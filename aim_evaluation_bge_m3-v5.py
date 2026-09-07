#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIM Challenge 2026 - 추천 근거성 정량 평가 스크립트 (BGE-M3 기반) [v5]
------------------------------------------------------------------
이 스크립트는 AIM Challenge 2026 정량평가 가이드에 안내된 BGE-M3 모델 기반의
'추천 근거성 평가 (Recommendation Grounding Evaluation)' 알고리즘을 구현한 평가 도구의 최종 무결점 버전입니다.

[v5 개선 사항]
1. pd.isna() 비-스칼라 입력 방어 고도화 (🔴 필수):
   - clean_val() 유틸리티에 입력값 자체가 DataFrame/Series/List 등의 컬렉션인 경우를 검사하는 __iter__ 방어 루틴을 장착해
     Ambiguous Truth Value 에러를 원천적이고 깔끔하게 해결했습니다.
2. CSV 로드 시 필수 컬럼 검증 강화 (🔴 필수):
   - 로드 직후 "place_id", "evidence_text_1", "evidence_text_2" 컬럼의 존재 여부를 강하게 검증하여
     필수 데이터 누락 시 즉각 유의미한 ValueError 예외를 던지도록 수정했습니다.
3. Faithfulness 데드 코드 제거 (🟡 권장):
   - evidence_list가 비어있을 때 사용하지 않고 남아있던 faith_ratio 할당 선언문을 정리했습니다.

[평가 지표]
1. 지역 추천지 유효성 (Validity Ratio, 20%): 추천한 place_id가 기준 데이터에 존재하는가?
2. 추천이유-근거 매칭도 (Grounding Score, 50%): 추천 이유가 5개 의미 축과 얼마나 부합하는가?
3. 설명 충실도 (Faithfulness Score, 30%): 각 문장이 실제 근거(evidence_text)에 의해 지지되는가? (임계값 0.58)
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def parse_sentences(text):
    """
    마침표(.), 느낌표(!), 물음표(?)를 기준으로 문장을 분리합니다.
    5자 이하의 짧은 조각은 문장 수에서 제외합니다.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # 문장 기호 기준 분리
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        # 5자 초과 문장만 인정
        if len(s_clean) > 5:
            sentences.append(s_clean)
            
    return sentences

def clean_val(val):
    """
    pandas nan, float nan, None, "nan" 등의 결측치와 비-스칼라(DataFrame, Series, List 등) 값들을
    안전하게 빈 문자열("")로 정제합니다.
    """
    if val is None:
        return ""
    
    # 컬렉션(DataFrame, Series, list, dict 등)이 유입되면 빈 문자열 반환 (문자열/바이트 제외)
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        return ""
        
    try:
        is_na = pd.isna(val)
        if hasattr(is_na, "__iter__"):
            return ""
        if is_na:
            return ""
    except Exception:
        pass
    
    val_str = str(val).strip()
    if val_str.lower() in ("nan", "none", "null"):
        return ""
    return val_str

class BGEM3Evaluator:
    def __init__(self, model_name="BAAI/bge-m3", device="cpu"):
        print(f"BGE-M3 모델 및 토크나이저 로드 중: {model_name} (디바이스: {device})")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("모델 로드 완료!")

    @torch.no_grad()
    def get_dense_embeddings(self, texts):
        """
        BGE-M3를 이용해 텍스트 리스트의 Dense 임베딩을 구하고 정규화(L2 normalize)합니다.
        """
        if not texts:
            return torch.zeros((0, 1024), device=self.device)
            
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=8192, 
            return_tensors='pt'
        ).to(self.device)
        
        model_output = self.model(**encoded_input)
        # BGE-M3는 CLS Pooling 사용
        embeddings = model_output[0][:, 0]
        # L2 Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def cosine_similarity(self, emb1, emb2):
        """
        두 정규화된 임베딩 행렬 간의 코사인 유사도를 계산합니다.
        """
        # emb1: (M, D), emb2: (N, D)
        return torch.matmul(emb1, emb2.T).cpu().numpy()

def evaluate(csv_path, json_path, evaluator):
    # 1. 파일 검증 및 로드
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"기준 CSV 파일을 찾을 수 없습니다: {csv_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"추천지 JSON 파일을 찾을 수 없습니다: {json_path}")
        
    df_ref = pd.read_csv(csv_path)
    # 컬럼 공백 제거 및 소문자 변환 대비
    df_ref.columns = [c.strip() for c in df_ref.columns]
    
    # 필수 컬럼 존재 여부 사전 체크
    REQUIRED_COLS = {"place_id", "evidence_text_1", "evidence_text_2"}
    missing = REQUIRED_COLS - set(df_ref.columns)
    if missing:
        raise ValueError(f"기준 CSV 파일에 필수 컬럼이 누락되었습니다: {missing}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        rec_data = json.load(f)
        
    recommendations = rec_data.get("recommendations", [])
    total_recs = len(recommendations)
    
    if total_recs == 0:
        print("평가 오류: 추천 장소가 JSON 파일에 존재하지 않습니다.")
        return
        
    # 기준 데이터 인덱싱
    ref_places = {}
    for _, row in df_ref.iterrows():
        pid = str(row['place_id']).strip()
        # 빈 place_id 인덱싱 방지 가드
        if pid and pid.lower() != "nan":
            ref_places[pid] = row
        
    print(f"\n--- [AIM Challenge 2026 정량평가 검증 시작 - v5] ---")
    print(f"기준 데이터 장소 수: {len(ref_places)}개")
    print(f"제출 추천 장소 수: {total_recs}개")
    
    # 2. 지표별 스코어 계산 준비
    valid_count = 0
    valid_places_info = []
    
    # 추천인 및 장소 아이디 존재 확인 (중복 제거 검증 포함)
    seen_place_ids = set()
    
    for rec in recommendations:
        pid = str(rec.get("place_id") or "").strip()
        pname_json = rec.get("place_name") or ""  # pname None 가드 반영
        
        # None 폴백이 비활성화되는 버그 수정 (or 연산 적용)
        reason = (
            rec.get("recommend_reason")
            or rec.get("recommendation_reason")
            or rec.get("reason")
            or ""
        ).strip()
        
        # 중복 체크 로직 동기화 버그 수정 및 빈 place_id 가드 추가
        is_duplicate = pid in seen_place_ids if pid else False
        is_valid = bool(pid) and (pid in ref_places) and not is_duplicate
        if is_valid:
            valid_count += 1
            seen_place_ids.add(pid)
            
        valid_places_info.append({
            "place_id": pid,
            "place_name": pname_json,
            "reason": reason,
            "is_valid": is_valid,
            "duplicate": is_duplicate
        })

    # ① 지역 추천지 유효성 (Validity Ratio, 20% 가중치)
    validity_ratio = valid_count / total_recs if total_recs > 0 else 0
    # 정규화: (validity_ratio - 0.50) / 0.50 * 100 (0점 미만은 0점 처리)
    if validity_ratio < 0.50:
        validity_score = 0.0
    else:
        validity_score = ((validity_ratio - 0.50) / 0.50) * 100.0
    validity_score = min(max(validity_score, 0.0), 100.0)
    
    print(f"\n[1] 지역 추천지 유효성: {valid_count}/{total_recs}개 유효 (유효율: {validity_ratio*100:.1f}%)")
    print(f"   => 유효성 정규화 점수: {validity_score:.2f} / 100.0")

    # ② 의미 근거성 및 설명 충실도 평가 시작
    place_grounding_scores = []
    place_faithfulness_scores = []
    
    for item in valid_places_info:
        pid = item["place_id"]
        reason = item["reason"]
        is_valid = item["is_valid"]
        
        # 유효하지 않은 추천지나 중복 추천지는 무조건 0점 처리!
        if not is_valid:
            pname_invalid = clean_val(item.get("place_name")) or "알 수 없는 장소"
            print(f"장소 {pid} ({pname_invalid}) (무효/중복): 의미 근거성 및 충실도 0점 산입")
            place_grounding_scores.append(0.0)
            place_faithfulness_scores.append(0.0)
            continue
            
        ref_row = ref_places[pid]
        pname = clean_val(item.get("place_name")) or clean_val(ref_row.get("place_name")) or "이름 없음"
        
        # 빈 reason 조기 차단 가드 적용
        if not reason:
            print(f"장소 {pid} ({pname}): 추천이유 없음 -> 추천이유 평가 0점 처리")
            place_grounding_scores.append(0.0)
            place_faithfulness_scores.append(0.0)
            continue
            
        # 2문장 규칙 검증
        sentences = parse_sentences(reason)
        num_sentences = len(sentences)
        
        if num_sentences < 2:
            print(f"장소 {pid} ({pname}): 문장 수 미달 ({num_sentences}문장) -> 추천이유 평가 0점 처리")
            place_grounding_scores.append(0.0)
            place_faithfulness_scores.append(0.0)
            continue
            
        # 5문장 초과 시 앞 5문장만 인식
        eval_sentences = sentences[:5]
        num_eval = len(eval_sentences)
        
        # 근거 텍스트 추출 및 정제 (clean_val 가드 반영)
        evidence_1 = clean_val(ref_row.get("evidence_text_1"))
        evidence_2 = clean_val(ref_row.get("evidence_text_2"))
        
        # 객관/주관 근거 텍스트가 둘 다 부재할 경우 가드 처리
        if not evidence_1 and not evidence_2:
            print(f"장소 {pid} ({pname}): 근거 텍스트(evidence_text_1/2) 없음 -> 0점 처리")
            place_grounding_scores.append(0.0)
            place_faithfulness_scores.append(0.0)
            continue

        # 테마 합성 (summary + category) 및 nan/None 가드 반영
        summary = clean_val(ref_row.get("summary"))
        category = clean_val(ref_row.get("category"))
        
        theme_text = f"{summary} {category}".strip()
        if not theme_text:
            theme_text = clean_val(ref_row.get("embedding_text"))
                
        # theme_text 빈 값 가드 및 가이드라인에 맞춘 백업 폴백 라인 구축 (pname None 방어 완료)
        if not theme_text:
            theme_text = evidence_1 or evidence_2 or pname or ""
            
        # 4. 기타 보충문장들 (문장별 충실도 검증용)
        evidence_list = [evidence_1, evidence_2]
        for col in ["evidence_text_3", "evidence_text_4", "evidence_text_5"]:
            val = clean_val(ref_row.get(col))
            if val:
                evidence_list.append(val)
        evidence_list = [e for e in evidence_list if e] # 빈 문자열 제외
        
        # --- ② 추천이유-근거 매칭도 (Grounding Score) 계산 ---
        # 빈 텍스트를 배제한 효율적이고 비어있지 않은 임베딩 연산
        texts_to_embed = [reason]
        indices = {"reason": 0}
        
        if evidence_1:
            indices["evidence_1"] = len(texts_to_embed)
            texts_to_embed.append(evidence_1)
        if evidence_2:
            indices["evidence_2"] = len(texts_to_embed)
            texts_to_embed.append(evidence_2)
        if theme_text:
            indices["theme_text"] = len(texts_to_embed)
            texts_to_embed.append(theme_text)
            
        embeddings = evaluator.get_dense_embeddings(texts_to_embed)
        reason_emb = embeddings[indices["reason"] : indices["reason"] + 1]
        
        # 객관 축 계산
        if "evidence_1" in indices:
            ev1_emb = embeddings[indices["evidence_1"] : indices["evidence_1"] + 1]
            sim_obj = evaluator.cosine_similarity(reason_emb, ev1_emb)[0][0]
            n_obj = ((sim_obj - 0.40) / 0.28) * 100.0
            n_obj = min(max(n_obj, 0.0), 100.0)
            obj_display = f"{sim_obj:.3f}(정규화: {n_obj:.1f}점)"
        else:
            sim_obj = 0.0
            n_obj = 0.0
            obj_display = "N/A"
            
        # 주관 축 계산
        if "evidence_2" in indices:
            ev2_emb = embeddings[indices["evidence_2"] : indices["evidence_2"] + 1]
            sim_subj = evaluator.cosine_similarity(reason_emb, ev2_emb)[0][0]
            n_subj = ((sim_subj - 0.52) / 0.30) * 100.0
            n_subj = min(max(n_subj, 0.0), 100.0)
            subj_display = f"{sim_subj:.3f}(정규화: {n_subj:.1f}점)"
        else:
            sim_subj = 0.0
            n_subj = 0.0
            subj_display = "N/A"
            
        # 테마 축 계산
        if "theme_text" in indices:
            theme_emb = embeddings[indices["theme_text"] : indices["theme_text"] + 1]
            sim_theme = evaluator.cosine_similarity(reason_emb, theme_emb)[0][0]
            n_theme = ((sim_theme - 0.55) / 0.17) * 100.0
            n_theme = min(max(n_theme, 0.0), 100.0)
            theme_display = f"{sim_theme:.3f}(정규화: {n_theme:.1f}점)"
        else:
            sim_theme = 0.0
            n_theme = 0.0
            theme_display = "N/A"
            
        # 근거 매칭도 공식: 0.7 * max(n_객관, n_주관) + 0.3 * n_테마
        grounding_score = 0.70 * max(n_obj, n_subj) + 0.30 * n_theme
        place_grounding_scores.append(grounding_score)
        
        # --- ③ 설명 충실도 (Faithfulness) 계산 ---
        # evidence_list가 비어있더라도 Grounding 점수는 보존하고 Faithfulness 점수만 0점 처리 (v5: faith_ratio 데드코드 제거 완료)
        if not evidence_list:
            place_faithfulness_scores.append(0.0)
            print(f"장소 {pid} ({pname}):")
            print(f"   - 매칭도: 객관 {obj_display} / 주관 {subj_display} / 테마 {theme_display}")
            print(f"     => 최종 근거성 점수: {grounding_score:.2f}점")
            print(f"   - 충실도: 0/{num_eval} 문장 지지됨 (지지율: 0.0%) [근거 데이터 없음]")
        else:
            supported_count = 0
            
            # 문장들과 근거들 임베딩 연산
            sent_embs = evaluator.get_dense_embeddings(eval_sentences)
            ref_embs = evaluator.get_dense_embeddings(evidence_list)
            
            sim_matrix = evaluator.cosine_similarity(sent_embs, ref_embs)
            
            sent_details = []
            for s_idx, sent in enumerate(eval_sentences):
                max_sim = np.max(sim_matrix[s_idx])
                is_supported = max_sim >= 0.58
                if is_supported:
                    supported_count += 1
                sent_details.append(f"   - 문장 {s_idx+1}: '{sent[:25]}...' -> 최대 유사도: {max_sim:.4f} ({'✓ 지지됨' if is_supported else '✗ 지지 안됨'})")
                
            faith_ratio = supported_count / num_eval if num_eval > 0 else 0.0
            place_faithfulness_scores.append(faith_ratio)
            
            print(f"장소 {pid} ({pname}):")
            print(f"   - 매칭도: 객관 {obj_display} / 주관 {subj_display} / 테마 {theme_display}")
            print(f"     => 최종 근거성 점수: {grounding_score:.2f}점")
            print(f"   - 충실도: {supported_count}/{num_eval} 문장 지지됨 (지지율: {faith_ratio*100:.1f}%)")
            for detail in sent_details:
                print(detail)

    # 전체 장소에 대한 평균 계산
    avg_grounding = np.mean(place_grounding_scores) if place_grounding_scores else 0.0
    avg_faith_ratio = np.mean(place_faithfulness_scores) if place_faithfulness_scores else 0.0
    
    # 충실도 최종 정규화: (avg_faith_ratio - 0.33) / 0.67 * 100
    if avg_faith_ratio < 0.33:
        final_faithfulness_score = 0.0
    else:
        final_faithfulness_score = ((avg_faith_ratio - 0.33) / 0.67) * 100.0
    final_faithfulness_score = min(max(final_faithfulness_score, 0.0), 100.0)

    # 3. 종합 평가 결과 산출 (기본 예선 가중치 반영)
    # 추천 근거성 종합 점수 = 유효성 * 0.20 + 의미근거성 * 0.50 + 설명충실도 * 0.30
    total_recommend_grounding = (validity_score * 0.20) + (avg_grounding * 0.50) + (final_faithfulness_score * 0.30)
    
    print(f"\n--- [최종 추천 근거성 정량 평가 결과] ---")
    print(f"1. 지역 추천지 유효성 (20%): {validity_score:.2f} 점 (유효 추천지 비율: {validity_ratio*100:.1f}%)")
    print(f"2. 추천이유 근거 매칭도 (50%): {avg_grounding:.2f} 점 (장소별 평균)")
    print(f"3. 설명 충실도 정규화 (30%): {final_faithfulness_score:.2f} 점 (평균 문장 지지율: {avg_faith_ratio*100:.1f}%)")
    print(f"==================================================")
    print(f"★ 추천 근거성 최종 점수 (100점 만점): {total_recommend_grounding:.2f} 점")
    print(f"==================================================")
    
    # 등급 분류 (정량 평가 가이드 기준)
    if total_recommend_grounding >= 90.0:
        grade = "매우 우수 (S)"
    elif total_recommend_grounding >= 75.0:
        grade = "우수 (A)"
    elif total_recommend_grounding >= 60.0:
        grade = "보통 (B)"
    elif total_recommend_grounding >= 40.0:
        grade = "부족 (C)"
    else:
        grade = "매우 부족 (D)"
    print(f"최종 정량 점수 등급: {grade}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIM Challenge 2026 추천 근거성 자동 채점기 - v5")
    parser.add_argument("--csv_path", type=str, required=True, help="지역 기준 데이터 CSV 파일 경로")
    parser.add_argument("--json_path", type=str, required=True, help="추천 결과 JSON 파일 경로")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3", help="Hugging Face BGE-M3 모델 이름 또는 로컬 경로")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="연산에 사용할 디바이스")
    
    args = parser.parse_args()
    
    try:
        evaluator = BGEM3Evaluator(model_name=args.model_name, device=args.device)
        evaluate(args.csv_path, args.json_path, evaluator)
    except Exception as e:
        print(f"\n평가 실행 중 오류가 발생했습니다: {e}")
