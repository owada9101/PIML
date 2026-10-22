"""Check saved trial artifacts and write a Japanese report with measured numbers."""

import argparse
import json
from pathlib import Path

import numpy as np

from .dynamics import Parameters, rk4_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    args = parser.parse_args()
    out = args.trial / "evaluation"
    report = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    resolution = json.loads((out / "resolution_checks.json").read_text(encoding="utf-8"))
    assert len(resolution["closed_loop"]) == 12
    assert all(r["same_outcome"] for r in resolution["closed_loop"])
    for seed in range(3):
        with np.load(args.trial / f"pino_seed{seed}" / "prediction.npz") as a, np.load(args.trial / f"fno_seed{seed}" / "prediction.npz") as b:
            for key in ["initial", "force", "target"]:
                np.testing.assert_array_equal(a[key], b[key])
    runs = [np.load(path) for pattern in ["controls_*.npz", "grid_*.npz"] for path in sorted(out.glob(pattern))]
    replay = np.stack([r["states"][0] for r in runs])
    max_error = np.zeros(4)
    for i in range(max(len(r["forces"]) for r in runs)):
        active = [j for j, r in enumerate(runs) if i < len(r["forces"])]
        applied = np.stack([runs[j]["actual_total_forces"][i] for j in active])
        replay[active] = rk4_step(replay[active], applied[:, 0], applied[:, 1], 0.05, substeps=20)
        reference = np.stack([runs[j]["states"][i + 1] for j in active])
        max_error = np.maximum(max_error, np.abs(replay[active] - reference).max(axis=0))
    for r in runs:
        assert np.isfinite(r["states"]).all()
        assert np.max(np.abs(r["forces"])) <= 10.00001
        assert len(r["times"]) == len(r["forces"]) + 1
    checks = dict(saved_control_runs=len(runs), finite_states=True, control_limit_verified=True,
                  paired_test_draws_identical=True,
                  rk4_half_step_replay_max_abs_state_error=max_error.tolist(),
                  note="Recorded controls replayed with 20 rather than 10 RK4 substeps; control plans not recomputed.")
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    for r in runs:
        r.close()

    def values(kind, case, metric="scaled_rmse"):
        return np.array([r[metric] for r in report["prediction"] if r["kind"] == kind and r["case"] == case])

    cases = [("学習範囲内の未使用条件", "in_distribution"),
             ("初期傾き20〜25度", "angle_20_to_25_deg"),
             ("力の節点振幅10〜16 N", "force_knots_10_to_16_N"),
             ("短いパルス波形", "pulse_shape"),
             ("初期状態を粗く丸める", "quantized_initial_state")]
    pin = values("pino", "in_distribution").mean()
    fno = values("fno", "in_distribution").mean()
    lines = ["# 倒立振子PINO：初回学習・評価結果（2026-10-22）", "",
             f"学習範囲内の未使用128条件では、PINOの無次元RMSEは平均 **{pin:.6f}**、物理残差なしのFNOは **{fno:.6f}**。PINOの誤差が **{(1-pin/fno)*100:.1f}%小さい**結果でした。3学習シードのそれぞれでPINOが改善しています。", "",
             "これは物理モデルを組み込んだ予測器の学習実験です。操作力は、固定した学習済み予測器を使うMPCが別途最適化します。操作方策を直接学習したものではありません。スライドはまだ作成していません。", "",
             "## 学習条件", "",
             "- 台車1 kg、振子0.1 kg、長さ0.5 m、重力9.81 m/s²。摩擦なしの非線形モデル。",
             "- 入力：初期状態 [位置, 速度, 傾き, 角速度] と0.4秒間の力の波形。出力：同区間の状態軌道。",
             "- 初期状態の学習範囲：位置±0.4 m、速度±1 m/s、傾き±0.3 rad（約17.2度）、角速度±1.5 rad/s。力は9節点で各±10 N、節点間は線形補間。",
             "- 512軌道で学習、64軌道でモデル選択。1,500更新、バッチ32。乱数シード0/1/2。シードごとに同じ初期重み・データ・バッチ列でPINO/FNOを対にした。",
             "- 予測は65時点、教師データの損失は17時点、物理残差は65時点。FNOの幅32、周波数モード12、3層。",
             "- 両者に初期条件と積分構造を共通に組み込み、比較する差は物理残差の重み（PINO=1、FNO=0）。FNO側にも構造上の物理的事前知識はある。",
             "- データ損失：状態誤差を [l, √(gl), 1, √(g/l)] で割ったMSE。物理損失：予測加速度と運動方程式から求めた加速度との差を [g, g/l] で割ったMSE。",
             "- 共通評価用の128条件は別乱数90210等で生成し、全モデルへ同じ条件を入力。モデル選択・再学習には使わない。",
             f"- 環境：Python {report['environment']['python']}、PyTorch {report['environment']['torch']}、Windows CPU・2スレッド。比較実験は順番に実行。", "",
             "| シード | PINO学習時間 | FNO学習時間 | PINO採用更新 | FNO採用更新 |", "|---|---:|---:|---:|---:|"]
    for seed in range(3):
        rows = [json.loads((args.trial / f"{k}_seed{seed}" / "training.json").read_text())["metrics"] for k in ["pino", "fno"]]
        lines.append(f"| {seed} | {rows[0]['training_seconds']:.1f}秒 | {rows[1]['training_seconds']:.1f}秒 | {rows[0]['best_step']} | {rows[1]['best_step']} |")
    lines += ["", "## 予測と分布外評価", "",
              f"共通の未使用条件に対する学習前のRMSEは **{report['untrained']['scaled_rmse']:.6f}**。この初期モデルは加速度出力がゼロなので、初期速度のまま進む予測です。", "",
              "数値は3シードの平均±標準偏差（ddof=0）。RMSEは4状態と65時点を平均した無次元量で、百分率の誤差ではありません。", "",
              "| 評価条件（各128軌道） | PINO | 物理残差なしFNO |", "|---|---:|---:|"]
    for label, case in cases:
        a, b = values("pino", case), values("fno", case)
        lines.append(f"| {label} | {a.mean():.6f} ± {a.std():.6f} | {b.mean():.6f} ± {b.std():.6f} |")
    lines += ["", f"学習範囲内での傾きのRMSEは、PINO平均 {np.rad2deg(values('pino', 'in_distribution', 'rmse_si')[:, 2]).mean():.3f}度、FNO平均 {np.rad2deg(values('fno', 'in_distribution', 'rmse_si')[:, 2]).mean():.3f}度でした。",
              f"同条件の無次元物理残差RMSはPINO {values('pino', 'in_distribution', 'physics_rms').mean():.5f}、FNO {values('fno', 'in_distribution', 'physics_rms').mean():.5f}でした。", "",
              "分布外でも今回の比較では改善が残りました。ただし強い力では両者とも誤差が増え、任意の分布外入力への保証にはなりません。パルスは65時点で表現し、端は1セル（6.25 ms）で立ち上げ・立ち下げる近似です。真の不連続入力や観測格子より短い衝撃への検証は含みません。", "",
              "粗い初期状態の予測は、丸める前の真の状態から始まる軌道との比較です。観測誤差とモデル誤差の合計を含みます。", "",
              "![学習曲線と未使用条件の評価](evaluation/learning_and_generalization.png)", "",
              "## 倒立維持", "",
              "初期位置0.2 m・傾き10度、速度0。1.5〜1.6秒に台車へ2 Nの外力。5秒間のシミュレーション。基本実験では外力を制御器へ事前に教えず、状態観測から立て直します。", "",
              "MPCの予測時間0.4秒、操作更新50 ms、操作力±10 N、LBFGS最大15反復。Q=diag(4,1,80,2)、R=0.1、終端重みは局所LQRのP。", "",
              "倒立維持は全区間で傾き±30度以内・位置±2.4 m以内。終端収束は加えて位置±0.1 m、速度±0.1 m/s、傾き±2度、角速度±0.1 rad/s未満。", "",
              "| 方法 | 倒立維持 | 終端収束 |", "|---|---:|---:|"]
    for kind in ["uncontrolled", "lqr", "pino", "fno"]:
        rows = [r for r in report["controls"] if r["case"] == "nominal" and r["kind"] == kind]
        lines.append(f"| {kind} | {sum(r['balanced_entire_run'] for r in rows)}/{len(rows)} | {sum(r['settled_at_end'] for r in rows)}/{len(rows)} |")
    lines += ["", "学習モデルの状態予測を実測扱いせず、独立したRK4数値積分で実際の状態を進めています。", "",
              "![基本条件の制御比較](evaluation/control_comparison.png)", "", "[PINOの動作GIF](evaluation/pino_control.gif)", "",
              "## 粗い情報による制御", "",
              "seed 0・同じ基本初期条件。観測を位置1 cm、速度0.05 m/s、角度1度、角速度0.1 rad/s刻みへ丸めました。この実験では現在の外力推定値を受け取り、予測区間で一定と仮定します。将来のパルス開始・終了は渡しません。推定バイアスはパルス外にも加わります。", "",
              "| 方法 | 外力推定バイアス | 倒立維持 | 終端収束 | 終端位置 |", "|---|---:|---|---|---:|"]
    for r in report["controls"]:
        if r["case"] == "coarse":
            bias = f"{r['disturbance_bias_N']:+g} N" if r["kind"] != "lqr" else "不使用"
            lines.append(f"| {r['kind']} | {bias} | {'成功' if r['balanced_entire_run'] else '失敗'} | {'成功' if r['settled_at_end'] else '未達'} | {r['final_state'][0]:.3f} m |")
    lines += ["", "基本実験と粗い情報の実験では外力情報の有無も変わります。粗さだけの効果を切り分ける完全な要因比較ではありません。推定バイアスがある場合、倒立維持と原点への収束を区別する必要があります。", "",
              "## 初期傾き・外力の25条件", "",
              "seed 0。傾き −20/−10/0/10/20度 × 外力 −6/−3/0/3/6 N。位置0.2 m、正確な状態観測、外力は未知。他の条件は基本実験と同じ。", "",
              "| 方法 | 倒立維持 | 終端収束 |", "|---|---:|---:|"]
    for kind in ["pino", "fno", "lqr"]:
        rows = [r for r in report["grid"] if r["kind"] == kind]
        lines.append(f"| {kind} | {sum(r['balanced_entire_run'] for r in rows)}/{len(rows)} | {sum(r['settled_at_end'] for r in rows)}/{len(rows)} |")
    lines += ["", "±20度は学習初期角度の範囲外です。他の条件でも途中の状態が学習範囲外へ出る可能性があるため、観測状態の範囲外割合・最適化後の合力候補の最大値・実際の合力の最大値をJSONに記録しています。最適化途中の全候補の範囲は集計していません。", "",
              "![25条件の結果](evaluation/control_grid.png)", "", "## 計算時間", "",
              "PINOの0.4秒軌道の推論単体は、ウォームアップ10回後・100回計測の中央値が各シードで約 " + "/".join(f"{r['median_ms']:.2f}" for r in report["inference"] if r["kind"] == "pino") + " msでした。", "",
              "以下は基本条件のMPC全体。ウォーム値は最初の制御呼び出しを除外した99回から計算。", "",
              "| 方法・シード | 初回 | 中央値 | 95%点 | 最大 | 50 ms超過率 |", "|---|---:|---:|---:|---:|---:|"]
    for r in report["controls"]:
        if r["case"] == "nominal" and r["kind"] in ["pino", "fno"]:
            lines.append(f"| {r['kind']} {r['seed']} | {r['first_planning_ms']:.1f} ms | {r['warm_median_planning_ms']:.1f} ms | {r['warm_p95_planning_ms']:.1f} ms | {r['warm_max_planning_ms']:.1f} ms | {r['warm_deadline_miss_fraction']:.1%} |")
    lines += ["", "予測は速くても、操作決定には最適化の反復が必要です。今回のMPCは50 ms周期を安定して満たしていません。計算遅延を物理シミュレーションに入れていないため、この倒立成功を実時間制御の成功とは扱えません。", "",
              "## 検証と残る範囲", "",
              f"- 保存した{len(runs)}制御試行の状態が有限で、操作力が±10 N以内であることを確認。",
              "- RK4の内部刻みを半分（10→20分割）にし、各観測状態から1制御区間を再積分。最大状態差 [m, m/s, rad, rad/s] は " + ", ".join(f"{v:.2e}" for v in resolution['local_half_step_max_abs_state_difference']) + "。",
              "- さらに操作を毎回最適化し直す閉ループで12試行を再実行：基本条件のPINO/FNO各3シード、（−20度, −6 N）・（20度, 6 N）のPINO/FNO/LQR各1試行。倒立維持・終端収束の判定は全12試行で一致した。",
              "- 保存操作だけを5秒間再生する別チェックでは状態差が増幅した（最大角度差 " + f"{max_error[2]:.2f}" + " rad）。不安定な倒立振子で観測フィードバックを外した再生は閉ループの精度判定にならないため、上記の1区間比較と閉ループ再実行を用いた。生の再生差も checks.json に残してある。",
              "- 既存の物理整合性テスト5件が通過（平衡・力の符号、運動方程式、エネルギー保存、局所LQR安定性、積分構造と入力勾配）。",
              "- 学習範囲内だけでなく、今回の分布外予測でも物理残差ありの方が良好。ただしこの事実だけで制御成功率や計算速度の優位性は主張しない。",
              "- 数値モデルMPCとの比較、粗い情報の9条件×要因別比較、全3シードでの25条件制御、遅延を含む制御、未知の質量・摩擦は未実施。今回は初回学習の確認に絞った。", "",
              "## 保存物と再実行", "",
              "- 各 `pino_seed*/`・`fno_seed*/`：選択した重み `model.pt`、全学習ログ `training.json`、各シードの未使用データ予測。",
              "- `evaluation/`：共通評価データ・全モデルの予測・各制御軌道のNPZ、集計JSON、図とGIF。",
              "- `environment.txt`：実行環境のパッケージ一覧。過去の `outputs/pino` 等は上書きしていない。", "",
              "プロジェクトルートで次を実行すると、新しい出力先に同じ設定で学習・評価する。", "", "```powershell",
              ".\\Example\\cartpole_pino\\run_trial.ps1", "```", ""]
    (args.trial / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(args.trial / "RESULTS.md")


if __name__ == "__main__":
    main()
