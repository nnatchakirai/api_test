#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIM Challenge 2026 - 대규모 자동 생성 및 일괄 채점 파이프라인 (Batch Auto-Evaluator)
----------------------------------------------------------------------------------
무작위 여행자 시나리오를 생성 -> 실제 FastAPI(/recommend) 에이전트 호출 -> v5 채점기로 자동 채점
"""

import os
import re
import json
import random
import argparse
import subprocess
import sys
import pandas as pd
import requests

# ================= [CONFIG] 실제 서버 스펙에 맞게 수정하세요 =================
API_URL = "http://localhost:8000/recommend"   # app.py의 엔드포인트
API_TIMEOUT = 180                              # Ollama 로컬 생성이 느릴 수 있어 넉넉히
# ============================================================================

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
    companion = random.choice(COMPANIONS)
    pref_set = random.choice(PREFERENCES)
    avoid_set = random.choice(AVOIDS)

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
    app.py의 /recommend 엔드포인트를 호출합니다.
    - 요청: {"preference": "<자연어 취향 문장>"}
    - 응답: [{"place_id":..., "place_name":..., "recommend_reason":...}, ...] (리스트 그대로)
    """
    try:
        resp = requests.post(API_URL, json={"preference": scenario["user_prompt"]}, timeout=API_TIMEOUT)
        resp.raise_for_status()
        result_list = resp.json()

        if not isinstance(result_list, list):
            print(f"  -> 예상과 다른 응답 형식: {type(result_list)} (리스트가 아님)")
            return {"recommendations": []}

        return {"recommendations": result_list}

    except Exception as e:
        print(f"  -> 에이전트 호출 실패: {e}")
        return {"recommendations": []}

def run_single_evaluation(csv_path, json_path, evaluator_script="aim_evaluation_bge_m3-v5.py"):
    cmd = [
        sys.executable, evaluator_script,
        "--csv_path", csv_path,
        "--json_path", json_path,
        "--device", "mps"
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        stdout = result.stdout

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

        recommend_result = call_your_recommendation_agent(scenario, args.csv_path)

        with open(temp_json_path, 'w', encoding='utf-8') as f:
            json.dump(recommend_result, f, ensure_ascii=False, indent=2)

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

    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)

    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(args.output_report, index=False, encoding="utf-8-sig")

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
