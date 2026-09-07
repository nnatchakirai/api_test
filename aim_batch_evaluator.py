#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIM Challenge 2026 - 대규모 자동 생성 및 일괄 채점 파이프라인 (Batch Auto-Evaluator)
----------------------------------------------------------------------------------
이 스크립트는 매번 서버에 접속해 JSON 파일을 개별적으로 받고 수동 채점하는 번거로움을 해결하기 위해
제작되었습니다. 

[작동 프로세스]
1. 무작위 여행자 시나리오(성향, 동반객, 선호/비선호 태그)를 자동으로 다양하게 생성합니다.
2. 사용자가 연동한 본인의 'AI 추천 Agent'를 구동하여 JSON 결과를 실시간으로 받아옵니다.
3. 우리가 제작한 최신 무결점 채점기(v5)를 서브프로세스로 자동 실행하여 결과를 실시간 채점합니다.
4. 모든 실행 결과를 종합하여 평균 점수, S등급 달성률, 주요 감점 사유 등이 담긴 일괄 리포트(CSV/MD)를 생성합니다.
"""

import os
import re
import json
import random
import argparse
import subprocess
import pandas as pd

# 무작위 프롬프트/시나리오 생성용 풀
COMPANIONS = ["가족", "연인(데이트)", "친구들", "혼자(솔로 여행)"]
PREFERENCES = [
    ["역사", "전통", "고즈넉함", "문화유적"],
    ["액티비티", "레저", "이색체험", "물놀이"],
    ["자연경관", "힐링", "산책", "조용함"],
    ["미식", "로컬맛집", "야시장", "지역먹거리"],
    ["야경", "야간나들이", "전망", "일몰/노을"]
]
AVOIDS = [
    ["액티비티", "번화가", "쇼핑"],
    ["도심", "시끄러운곳", "실내전시"],
    ["역사답사", "등산", "피크닉"],
    ["낮활동", "물놀이", "사색"]
]

def generate_random_scenario():
    """
    무작위 여행자 성향 및 프롬프트 생성
    """
    companion = random.choice(COMPANIONS)
    pref_set = random.choice(PREFERENCES)
    avoid_set = random.choice(AVOIDS)
    
    # 중복 제거
    avoid_set = [a for e in avoid_set for a in [e] if a not in pref_set]
    if not avoid_set:
        avoid_set = ["쇼핑", "복잡한곳"]

    scenario = {
        "companion": companion,
        "preference_tags": pref_set,
        "avoid_tags": avoid_set,
        "user_prompt": f"이번 주말에 {companion}과 함께 김해로 여행을 떠나려고 해. "
                      f"우리는 특히 {', '.join(pref_set)} 성향의 장소를 선호하고, "
                      f"{', '.join(avoid_set)} 같은 분위기나 활동은 피하고 싶어. "
                      f"우리의 취향에 딱 맞는 최적의 여행 코스 3~5곳을 엄선해서 추천해줘."
    }
    return scenario

def call_your_recommendation_agent(scenario, csv_path):
    """
    [🚨 필독: 사용자 연동 구간]
    이 함수 내부에 개발하신 AI Agent 또는 LLM 호출 API(OpenAI, Claude, LangChain 등) 코드를 연동하세요.
    현재는 흐름 검증을 위해 무작위로 추출하여 올바른 JSON 포맷을 반환하는 Mock 코드가 작성되어 있습니다.
    """
    # -------------------------------------------------------------
    # 💡 실제 구현 예시 (사용 시 주석을 해제하고 사용하세요):
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": "너는 AIM 2026 가이드라인을 준수하는 추천 에이전트야..."},
    #         {"role": "user", "content": f"기준데이터: {csv_path}\n성향: {scenario['user_prompt']}"}
    #     ]
    # )
    # return json.loads(response.choices[0].message.content)
    # -------------------------------------------------------------
    
    # 아래는 테스트 작동용 Mock 생성 코드입니다.
    try:
        df = pd.read_csv(csv_path)
        # 선호 태그와 매칭되는 장소 필터링 시도
        matches = []
        for _, row in df.iterrows():
            cat_list = str(row.get("category", "")).split(",")
            pref_tags_str = str(row.get("preference_tags", ""))
            
            # 무작위로 조금이라도 겹치면 후보 등록
            score = sum(1 for p in scenario["preference_tags"] if p in cat_list or p in pref_tags_str)
            matches.append((score, row))
            
        # 점수 순 정렬 후 상위 3~5개 선정
        matches.sort(key=lambda x: x[0], reverse=True)
        selected_rows = [item[1] for item in matches[:random.randint(3, 5)]]
        
        recommendations = []
        for row in selected_rows:
            # 2~3문장 추천 사유 무작위 합성 (환각 방지)
            ev1 = str(row.get("evidence_text_1", ""))
            ev2 = str(row.get("evidence_text_2", ""))
            pname = str(row.get("place_name", ""))
            
            reason = f"{pname}은 {ev1.replace('.', '')} 장소입니다. 또한 {ev2.replace('.', '')} 정취를 가득 선사하여 강력히 추천합니다."
            
            recommendations.append({
                "place_id": str(row.get("place_id")),
                "place_name": pname,
                "recommend_reason": reason
            })
            
        return {"recommendations": recommendations}
    except Exception as e:
        # 에러 발생 시 예외 안전 보장용 기본 구조 반환
        return {
            "recommendations": [
                {
                    "place_id": "GIMHAE_001",
                    "place_name": "수로왕릉",
                    "recommend_reason": "수로왕릉은 가락국 시조 수로왕의 무덤으로 역사적 깊이가 깊습니다. 평온한 왕릉공원을 산책하며 고즈넉한 가야의 정취를 차분하게 감상할 수 있어 가족과 방문하기에 더없이 좋습니다."
                }
            ]
        }

def run_single_evaluation(csv_path, json_path, evaluator_script="aim_evaluation_bge_m3-v5.py"):
    """
    Subprocess를 활용하여 작성된 v5 채점기를 안전하게 실행하고 점수를 파싱합니다.
    (메모리 오염 및 GPU OOM 방지)
    """
    cmd = [
        "python3", evaluator_script,
        "--csv_path", csv_path,
        "--json_path", json_path,
        "--device", "cpu" # 샌드박스 내부 검증용 (GPU가 있다면 cuda 지정 가능)
    ]
    
    # 만약 오프라인 환경에 BGE-M3가 없다면 run_evaluation_sim.py(시뮬레이터)를 대신 호출하도록 폴백
    if not os.path.exists(evaluator_script) and os.path.exists("run_evaluation_sim.py"):
        cmd[1] = "run_evaluation_sim.py"
    elif not os.path.exists(evaluator_script) and os.path.exists("/workspace/scratch/run_evaluation_sim.py"):
        cmd[1] = "/workspace/scratch/run_evaluation_sim.py"
        
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        stdout = result.stdout
        
        # 정규식을 활용한 평점 정보 스크랩
        validity_match = re.search(r"1\. 지역 추천지 유효성.*?: ([\d\.]+) 점", stdout)
        grounding_match = re.search(r"2\. 추천이유 근거 매칭도.*?: ([\d\.]+) 점", stdout)
        faithfulness_match = re.search(r"3\. 설명 충실도 정규화.*?: ([\d\.]+) 점", stdout)
        total_match = re.search(r"★ 추천 근거성 최종 점수.*?: ([\d\.]+) 점", stdout)
        grade_match = re.search(r"최종 정량 점수 등급: (.*)", stdout)
        
        scores = {
            "validity": float(validity_match.group(1)) if validity_match else 0.0,
            "grounding": float(grounding_match.group(1)) if grounding_match else 0.0,
            "faithfulness": float(faithfulness_match.group(1)) if faithfulness_match else 0.0,
            "total_score": float(total_match.group(1)) if total_match else 0.0,
            "grade": grade_match.group(1).strip() if grade_match else "N/A",
            "raw_log": stdout
        }
        return scores
    except Exception as e:
        print(f"채점 서브프로세스 실행 실패: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="AIM Challenge 2026 - Agent 자동 생성 및 일괄 채점기")
    parser.add_argument("--csv_path", type=str, required=True, help="지역 기준 데이터 CSV 파일 경로")
    parser.add_argument("--num_runs", type=str, default="5", help="테스트할 무작위 프롬프트 횟수")
    parser.add_argument("--eval_script", type=str, default="aim_evaluation_bge_m3-v5.py", help="v5 채점 스크립트 파일명")
    parser.add_argument("--output_report", type=str, default="batch_evaluation_summary.csv", help="종합 보고서 출력 CSV 경로")
    
    args = parser.parse_args()
    num_runs = int(args.num_runs)
    
    print(f"==================================================")
    print(f"🚀 AIM Challenge Batch Auto-Evaluator 가동 시작")
    print(f"- 대상 기준 데이터: {args.csv_path}")
    print(f"- 테스트 반복 횟수: {num_runs}회")
    print(f"==================================================\n")
    
    temp_json_path = "temp_batch_recommend.json"
    results = []
    
    for i in range(num_runs):
        print(f"👉 [{i+1}/{num_runs}] 차 여행자 시나리오 생성 중...")
        scenario = generate_random_scenario()
        print(f"   * 성향: {scenario['companion']} / 선호: {scenario['preference_tags']} / 기피: {scenario['avoid_tags']}")
        print(f"   * 에이전트 입력 프롬프트: \"{scenario['user_prompt'][:40]}...\"")
        
        # 1. Agent 구동 및 결과 수집
        recommend_result = call_your_recommendation_agent(scenario, args.csv_path)
        
        # 임시 JSON 파일 기록
        with open(temp_json_path, 'w', encoding='utf-8') as f:
            json.dump(recommend_result, f, ensure_ascii=False, indent=2)
            
        # 2. 실시간 자동 채점 실행
        scores = run_single_evaluation(args.csv_path, temp_json_path, args.eval_script)
        
        if scores:
            print(f"   => 🎯 채점 완료! 총점: {scores['total_score']:.2f} 점 (등급: {scores['grade']})")
            results.append({
                "run_id": i + 1,
                "companion": scenario["companion"],
                "preference_tags": ",".join(scenario["preference_tags"]),
                "avoid_tags": ",".join(scenario["avoid_tags"]),
                "num_places": len(recommend_result.get("recommendations", [])),
                "score_validity": scores["validity"],
                "score_grounding": scores["grounding"],
                "score_faithfulness": scores["faithfulness"],
                "total_score": scores["total_score"],
                "grade": scores["grade"]
            })
        else:
            print(f"   => ❌ 채점 오류 발생")
            
        print("-" * 50)
        
    # 임시 파일 정리
    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)
        
    # 3. 데이터프레임 빌드 및 파일 저장
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(args.output_report, index=False, encoding="utf-8-sig")
        
        # 대시보드 요약 정보 출력
        mean_score = df_res["total_score"].mean()
        max_score = df_res["total_score"].max()
        min_score = df_res["total_score"].min()
        s_rate = (df_res["total_score"] >= 90.0).sum() / len(df_res) * 100
        a_rate = (df_res["total_score"] >= 75.0).sum() / len(df_res) * 100
        
        print(f"\n==================================================")
        print(f"📊 대규모 일괄 채점 종합 결과 대시보드 ({num_runs}회 테스트 완료)")
        print(f"==================================================")
        print(f"1. 점수 통계:")
        print(f"   - 평균 점수: {mean_score:.2f} / 100.0 점")
        print(f"   - 최고 점수: {max_score:.2f} 점")
        print(f"   - 최저 점수: {min_score:.2f} 점")
        print(f"2. 등급 달성률:")
        print(f"   - 매우 우수 (S등급, 90점 이상) 달성률: {s_rate:.1f}%")
        print(f"   - 우수 이상 (A등급 이상, 75점 이상) 달성률: {a_rate:.1f}%")
        print(f"3. 상세 결과 CSV 파일이 생성되었습니다: '{args.output_report}'")
        print(f"==================================================\n")
    else:
        print("수집된 채점 결과가 없습니다.")

if __name__ == "__main__":
    main()
