
import base64
import io
import json
import pickle
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from supabase import create_client, Client

st.set_page_config(
    page_title="本草智研 · 云端 AI 工艺优化平台",
    page_icon="🌿",
    layout="wide",
)

FEATURES = [
    "first_ethanol_ratio_L_kg",
    "second_ethanol_ratio_L_kg",
    "first_extraction_time_h",
    "second_extraction_time_h",
    "loading_flow_BV_h",
    "water_wash_flow_BV_h",
    "water_wash_volume_BV",
    "elution_flow_BV_h",
    "elution_volume_BV",
]
FEATURE_LABELS = {
    "first_ethanol_ratio_L_kg": "第一次乙醇倍量 (L/kg)",
    "second_ethanol_ratio_L_kg": "第二次乙醇倍量 (L/kg)",
    "first_extraction_time_h": "第一次提取时间 (h)",
    "second_extraction_time_h": "第二次提取时间 (h)",
    "loading_flow_BV_h": "上样流量 (BV/h)",
    "water_wash_flow_BV_h": "水洗流量 (BV/h)",
    "water_wash_volume_BV": "水洗体积 (BV)",
    "elution_flow_BV_h": "洗脱流量 (BV/h)",
    "elution_volume_BV": "洗脱体积 (BV)",
}
BASELINE = {
    "first_ethanol_ratio_L_kg": 8.0,
    "second_ethanol_ratio_L_kg": 6.0,
    "first_extraction_time_h": 3.0,
    "second_extraction_time_h": 2.0,
    "loading_flow_BV_h": 2.0,
    "water_wash_flow_BV_h": 2.0,
    "water_wash_volume_BV": 2.0,
    "elution_flow_BV_h": 1.5,
    "elution_volume_BV": 5.0,
}
TARGETS = [
    "active_A_content_pct",
    "active_B_content_pct",
    "active_C_content_pct",
    "active_D_content_pct",
    "active_E_content_pct",
    "total_active_content_pct",
    "dry_extract_mass_kg",
    "ethanol_solution_consumption_L",
    "process_time_h",
    "production_cost_RMB",
]
TARGET_LABELS = {
    "active_A_content_pct": "活性成分 A 含量 (%)",
    "active_B_content_pct": "活性成分 B 含量 (%)",
    "active_C_content_pct": "活性成分 C 含量 (%)",
    "active_D_content_pct": "活性成分 D 含量 (%)",
    "active_E_content_pct": "活性成分 E 含量 (%)",
    "total_active_content_pct": "五种活性成分总含量 (%)",
    "dry_extract_mass_kg": "干燥提取物质量 (kg)",
    "ethanol_solution_consumption_L": "乙醇溶液消耗量 (L)",
    "process_time_h": "总生产时间 (h)",
    "production_cost_RMB": "生产成本 (元)",
}
LIMITS = {
    "active_A_content_pct": 4.0,
    "active_B_content_pct": 22.0,
    "active_C_content_pct": 2.0,
    "active_D_content_pct": 24.0,
    "active_E_content_pct": 4.0,
}
MAX_BULK_ROWS = 100

def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

@st.cache_resource
def get_public_client():
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

@st.cache_resource
def get_admin_client():
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

def require_public():
    client = get_public_client()
    if client is None:
        st.error("网站尚未完成数据库配置。请管理员在云端 Secrets 中填写 Supabase 配置。")
        st.stop()
    return client

def require_admin():
    client = get_admin_client()
    if client is None:
        st.error("管理员服务密钥尚未配置，管理功能暂不可用。")
        st.stop()
    return client

def fnum(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def show_metric_cards(approved_count, pending_count, model_meta):
    c1, c2, c3 = st.columns(3)
    c1.metric("正式实验记录", approved_count)
    c2.metric("待审核提交", pending_count if pending_count is not None else "—")
    c3.metric("已发布模型", model_meta.get("version", "暂无") if model_meta else "暂无")

def get_latest_model_meta():
    pub = get_public_client()
    if pub is None:
        return None
    try:
        res = (
            pub.table("models")
            .select("id,version,created_at,metrics,sample_count,feature_columns,target_columns,feature_bounds,artifact_path")
            .eq("is_published", True)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as exc:
        st.warning(f"读取模型信息失败：{exc}")
        return None

@st.cache_resource(show_spinner=False)
def load_model_artifact(model_id: str, artifact_path: str):
    admin = get_admin_client()
    if admin is None:
        raise RuntimeError("云端服务密钥未配置，无法读取共享模型。")
    raw = admin.storage.from_("model-artifacts").download(artifact_path)
    return pickle.loads(raw)

def fetch_approved(limit=500):
    pub = require_public()
    res = (
        pub.table("submissions")
        .select("id,created_at,submitter,experiment_id,source,inputs,outputs,notes")
        .eq("status", "approved")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []

def record_to_flat(row):
    flat = {
        "id": row.get("id"),
        "created_at": row.get("created_at"),
        "submitter": row.get("submitter"),
        "experiment_id": row.get("experiment_id"),
        "source": row.get("source"),
        "notes": row.get("notes"),
    }
    flat.update(row.get("inputs") or {})
    flat.update(row.get("outputs") or {})
    return flat

def template_dataframe():
    row = {"experiment_id": "baseline_001", "submitter": "", "source": "比赛软件模拟", **BASELINE}
    row.update({t: "" for t in TARGETS})
    row["notes"] = "请填入比赛软件对应计算结果"
    return pd.DataFrame([row])

def validate_submission(row):
    inputs = {}
    outputs = {}
    errors = []
    for f in FEATURES:
        v = fnum(row.get(f))
        if v is None or v <= 0:
            errors.append(f"{FEATURE_LABELS[f]}必须为大于0的数值")
        else:
            inputs[f] = v
    for t in TARGETS:
        v = fnum(row.get(t))
        if v is not None:
            outputs[t] = v
    if not outputs:
        errors.append("至少填写一个输出指标（例如 A–E 含量或总含量）")
    return inputs, outputs, errors

def insert_submission(payload):
    pub = require_public()
    return pub.table("submissions").insert(payload).execute()

def make_submission_payload(row, submitter, source):
    inputs, outputs, errors = validate_submission(row)
    if errors:
        return None, errors
    return {
        "submitter": (submitter or "").strip()[:80] or "匿名提交者",
        "experiment_id": str(row.get("experiment_id") or f"EXP-{uuid.uuid4().hex[:8]}")[:100],
        "source": (source or "比赛软件模拟")[:100],
        "inputs": inputs,
        "outputs": outputs,
        "notes": str(row.get("notes") or "")[:2000],
        "status": "pending",
    }, []

def admin_login_box():
    st.sidebar.markdown("---")
    st.sidebar.subheader("管理员入口")
    if st.session_state.get("admin_authenticated"):
        st.sidebar.success("管理员已登录")
        if st.sidebar.button("退出管理员"):
            st.session_state["admin_authenticated"] = False
            st.rerun()
        return True
    pwd = st.sidebar.text_input("管理员密码", type="password", key="admin_pwd")
    if st.sidebar.button("管理员登录"):
        configured = get_secret("ADMIN_PASSWORD")
        if configured and pwd == configured:
            st.session_state["admin_authenticated"] = True
            st.sidebar.success("登录成功")
            st.rerun()
        else:
            st.sidebar.error("密码错误或尚未配置管理员密码")
    return False

def fetch_admin_submissions(status):
    admin = require_admin()
    res = (
        admin.table("submissions")
        .select("*")
        .eq("status", status)
        .order("created_at", desc=False)
        .limit(500)
        .execute()
    )
    return res.data or []

def set_submission_status(submission_id, status, reason=""):
    admin = require_admin()
    admin.table("submissions").update({
        "status": status,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": "admin",
        "rejection_reason": reason[:1000],
    }).eq("id", submission_id).execute()

def build_training_frame(approved_rows):
    records = [record_to_flat(r) for r in approved_rows]
    df = pd.DataFrame(records)
    for c in FEATURES + TARGETS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def train_models(df, selected_targets, n_estimators=300, min_leaf=2):
    feature_cols = FEATURES.copy()
    clean_x = df.dropna(subset=feature_cols).copy()
    if len(clean_x) < 6:
        raise ValueError(f"完整输入参数记录只有 {len(clean_x)} 条，至少需要 6 条。")
    models = {}
    metrics = {}
    for target in selected_targets:
        subset = clean_x.dropna(subset=[target])
        if len(subset) < 6:
            continue
        X = subset[feature_cols].astype(float)
        y = subset[target].astype(float)
        if len(subset) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            test_model = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_leaf,
                random_state=42,
                n_jobs=-1,
            )
            test_model.fit(X_train, y_train)
            pred = test_model.predict(X_test)
            metrics[target] = {
                "MAE": float(mean_absolute_error(y_test, pred)),
                "R2": float(r2_score(y_test, pred)) if y_test.nunique() > 1 else None,
                "test_n": int(len(y_test)),
                "train_n": int(len(y_train)),
            }
        else:
            metrics[target] = {
                "MAE": None, "R2": None, "test_n": 0,
                "train_n": int(len(subset)),
                "note": "样本量不足，未进行独立测试集评价",
            }
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
        models[target] = model
    if not models:
        raise ValueError("没有输出字段达到至少6条有效记录。请检查数据。")
    bounds = {
        f: [float(clean_x[f].min()), float(clean_x[f].max())]
        for f in feature_cols
    }
    artifact = {
        "models": models,
        "feature_columns": feature_cols,
        "target_columns": list(models.keys()),
        "feature_bounds": bounds,
        "metrics": metrics,
        "training_count": int(len(clean_x)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return artifact

def publish_model(artifact):
    admin = require_admin()
    model_id = str(uuid.uuid4())
    version = datetime.now().strftime("v%Y%m%d-%H%M%S")
    artifact_path = f"{version}/{model_id}.pkl"
    binary = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    admin.storage.from_("model-artifacts").upload(
        artifact_path,
        binary,
        file_options={"content-type": "application/octet-stream", "upsert": "true"},
    )
    admin.table("models").insert({
        "id": model_id,
        "version": version,
        "artifact_path": artifact_path,
        "metrics": artifact["metrics"],
        "sample_count": artifact["training_count"],
        "feature_columns": artifact["feature_columns"],
        "target_columns": artifact["target_columns"],
        "feature_bounds": artifact["feature_bounds"],
        "is_published": False,
    }).execute()
    admin.table("models").update({"is_published": False}).eq("is_published", True).execute()
    admin.table("models").update({"is_published": True}).eq("id", model_id).execute()
    load_model_artifact.clear()
    return version

def check_quality(pred):
    checks = []
    missing = []
    for k, limit in LIMITS.items():
        if k in pred:
            checks.append((TARGET_LABELS[k], pred[k], limit, pred[k] >= limit))
        else:
            missing.append(k)
    total = pred.get("total_active_content_pct")
    if total is None and all(k in pred for k in LIMITS):
        total = sum(pred[k] for k in LIMITS)
    if total is not None:
        checks.append(("五种活性成分总含量 (%)", total, 75.0, total >= 75.0))
    complete = (len(missing) == 0 and total is not None)
    return checks, complete

def predict_one(artifact, values):
    x = pd.DataFrame([[values[f] for f in artifact["feature_columns"]]], columns=artifact["feature_columns"])
    return {k: float(m.predict(x)[0]) for k, m in artifact["models"].items()}

def page_header():
    st.title("🌿 本草智研")
    st.caption("中药提取纯化工艺云端 AI 智能优化平台")
    st.info("公开访问 · 任何获得链接的人均可提交实验数据 · 审核通过的数据进入共享训练集")

def main():
    page_header()
    pub = get_public_client()
    if pub is None:
        st.error("数据库尚未连接。请按部署说明配置 Supabase。")
        st.stop()

    admin_mode = admin_login_box()
    pages = ["工作台", "提交实验数据", "共享实验数据库", "质量预测", "工艺优化"]
    if admin_mode:
        pages.append("管理员审核与训练")
    page = st.sidebar.radio("功能导航", pages)

    try:
        approved = fetch_approved(limit=500)
    except Exception as exc:
        st.error(f"读取共享数据库失败：{exc}")
        st.stop()
    model_meta = get_latest_model_meta()

    if page == "工作台":
        pending_count = None
        if admin_mode:
            try:
                pending_count = len(fetch_admin_submissions("pending"))
            except Exception:
                pending_count = "—"
        show_metric_cards(len(approved), pending_count, model_meta or {})
        st.subheader("平台运行状态")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**当前共享数据**")
            st.write(f"已审核实验记录：**{len(approved)}** 条")
            st.write("公开访客可以提交新数据，提交后进入待审核队列。")
        with c2:
            st.markdown("**当前共享模型**")
            if model_meta:
                st.write(f"模型版本：**{model_meta['version']}**")
                st.write(f"训练记录：**{model_meta['sample_count']}** 条")
                st.write(f"预测指标：**{len(model_meta['target_columns'])}** 个")
            else:
                st.warning("尚未发布模型。管理员审核足够的实验数据后可训练并发布。")
        st.markdown("### 使用流程")
        st.markdown("1. 从比赛软件导出每组工艺参数和计算结果。\n2. 在「提交实验数据」上传 CSV 或逐条录入。\n3. 管理员审核通过后，数据进入共享训练集。\n4. 管理员训练并发布模型，所有访问者即可使用预测和优化功能。")
        st.warning("公开提交意味着任何人都可以提交数据。为保护模型质量，待审核数据不会自动参与训练。请勿提交个人敏感信息、账号密码或与实验无关的文件。")

    elif page == "提交实验数据":
        st.subheader("提交实验数据")
        st.write("所有访问者均可提交。每条记录需要包含9个工艺输入参数，并至少提供一个输出指标。")
        st.download_button(
            "下载 CSV 数据模板",
            data=template_dataframe().to_csv(index=False).encode("utf-8-sig"),
            file_name="本草智研_实验数据提交模板.csv",
            mime="text/csv",
        )
        tab1, tab2 = st.tabs(["逐条提交", "批量 CSV 提交"])
        with tab1:
            with st.form("single_submit"):
                submitter = st.text_input("提交者 / 团队名称（可选）")
                exp_id = st.text_input("实验编号（可选）")
                source = st.selectbox("数据来源", ["比赛软件模拟", "实验实测", "文献数据", "其他"])
                st.markdown("**工艺输入参数**")
                input_cols = st.columns(3)
                values = {}
                for i, f in enumerate(FEATURES):
                    with input_cols[i % 3]:
                        values[f] = st.number_input(
                            FEATURE_LABELS[f],
                            min_value=0.000001,
                            value=float(BASELINE[f]),
                            format="%.4f",
                            key=f"single_{f}",
                        )
                st.markdown("**计算结果 / 检测结果**")
                output_cols = st.columns(3)
                outputs = {}
                for i, t in enumerate(TARGETS):
                    with output_cols[i % 3]:
                        raw = st.text_input(TARGET_LABELS[t] + "（可留空）", key=f"single_out_{t}")
                        outputs[t] = fnum(raw)
                notes = st.text_area("备注（可选）", max_chars=2000)
                consent = st.checkbox("确认数据真实、来源清楚，且不包含个人敏感信息。", value=False)
                submitted = st.form_submit_button("提交到共享数据库", type="primary")
            if submitted:
                if not consent:
                    st.error("请先确认数据来源及隐私声明。")
                elif not any(v is not None for v in outputs.values()):
                    st.error("至少填写一个输出指标。")
                else:
                    row = {**values, **outputs, "experiment_id": exp_id, "notes": notes}
                    payload, errors = make_submission_payload(row, submitter, source)
                    if errors:
                        st.error("；".join(errors))
                    else:
                        try:
                            insert_submission(payload)
                            st.success("提交成功！记录已进入共享数据库的待审核队列。审核通过后将加入正式训练集。")
                        except Exception as exc:
                            st.error(f"提交失败：{exc}")

        with tab2:
            st.write("上传 CSV 模板格式的数据。每次最多提交100条。")
            file = st.file_uploader("选择 CSV 文件", type=["csv"], key="bulk_csv")
            bulk_submitter = st.text_input("提交者 / 团队名称（可选）", key="bulk_submitter")
            if file is not None:
                try:
                    df = pd.read_csv(file)
                    st.dataframe(df.head(10), use_container_width=True)
                    if st.button("提交批量数据", type="primary"):
                        if len(df) > MAX_BULK_ROWS:
                            st.error(f"单次最多提交 {MAX_BULK_ROWS} 行，请拆分文件。")
                        else:
                            missing = [f for f in FEATURES if f not in df.columns]
                            if missing:
                                st.error("缺少输入字段：" + "、".join(missing))
                            else:
                                payloads, errors = [], []
                                for idx, row in df.iterrows():
                                    payload, row_errors = make_submission_payload(
                                        row.to_dict(),
                                        bulk_submitter,
                                        str(row.get("source") or "比赛软件模拟"),
                                    )
                                    if row_errors:
                                        errors.append(f"第{idx+2}行：" + "；".join(row_errors))
                                    else:
                                        payloads.append(payload)
                                if errors:
                                    st.error("发现无效记录，未提交本批数据。前10条错误：\n" + "\n".join(errors[:10]))
                                elif payloads:
                                    insert_submission(payloads)
                                    st.success(f"成功提交 {len(payloads)} 条记录，已进入待审核队列。")
                except Exception as exc:
                    st.error(f"CSV读取失败：{exc}")

    elif page == "共享实验数据库":
        st.subheader("共享实验数据库")
        st.caption("此处仅展示管理员审核通过的记录；待审核或被拒绝的记录不会公开显示。")
        if approved:
            flat = pd.DataFrame([record_to_flat(r) for r in approved])
            display_cols = ["created_at", "submitter", "experiment_id", "source"] + FEATURES + [t for t in TARGETS if t in flat.columns]
            display_cols = [c for c in display_cols if c in flat.columns]
            st.dataframe(flat[display_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "导出已审核共享数据 CSV",
                data=flat.to_csv(index=False).encode("utf-8-sig"),
                file_name="本草智研_已审核共享实验数据.csv",
                mime="text/csv",
            )
        else:
            st.info("目前还没有审核通过的实验记录。欢迎提交第一批数据。")

    elif page == "质量预测":
        st.subheader("共享 AI 模型 · 质量预测")
        if not model_meta:
            st.warning("尚未发布模型。请等待管理员审核数据并训练模型。")
        else:
            st.success(f"当前使用共享模型：{model_meta['version']} · 训练记录 {model_meta['sample_count']} 条")
            try:
                artifact = load_model_artifact(model_meta["id"], model_meta["artifact_path"])
                cols = st.columns(3)
                values = {}
                for i, f in enumerate(artifact["feature_columns"]):
                    lo, hi = artifact["feature_bounds"][f]
                    if hi <= lo:
                        hi = lo + 0.0001
                    default = min(max(BASELINE[f], lo), hi)
                    with cols[i % 3]:
                        values[f] = st.number_input(
                            FEATURE_LABELS[f],
                            min_value=float(lo),
                            max_value=float(hi),
                            value=float(default),
                            format="%.4f",
                            key=f"pred_{f}",
                        )
                if st.button("预测产品质量", type="primary"):
                    pred = predict_one(artifact, values)
                    out = pd.DataFrame([{"预测指标": TARGET_LABELS.get(k,k), "预测值": v} for k,v in pred.items()])
                    st.dataframe(out, use_container_width=True, hide_index=True)
                    checks, complete_quality = check_quality(pred)
                    if checks:
                        st.markdown("**质量标准检查**")
                        check_df = pd.DataFrame([
                            {"项目": name, "预测值 (%)": round(value, 4), "标准下限 (%)": limit, "判断": "达标" if ok else "未达标"}
                            for name, value, limit, ok in checks
                        ])
                        st.dataframe(check_df, use_container_width=True, hide_index=True)
                        if not complete_quality:
                            st.warning("当前模型没有覆盖全部五种单体成分及总含量指标，暂时不能判断整体质量是否达标。")
                        elif all(x[3] for x in checks):
                            st.success("模型预测值满足当前录入的全部质量约束。")
                        else:
                            st.warning("模型预测值未完全满足质量约束。")
                    st.caption("预测结果仅用于筛选候选方案，不等同于比赛软件结果或实际质量检测。")
            except Exception as exc:
                st.error(f"加载共享模型失败：{exc}")

    elif page == "工艺优化":
        st.subheader("共享 AI 模型 · 工艺参数优化")
        if not model_meta:
            st.warning("尚未发布模型。")
        else:
            try:
                artifact = load_model_artifact(model_meta["id"], model_meta["artifact_path"])
                target_options = artifact["target_columns"]
                objective = st.selectbox("优化目标", target_options, format_func=lambda x: TARGET_LABELS.get(x,x))
                direction = st.radio("优化方向", ["最大化", "最小化"], horizontal=True)
                n_candidates = st.select_slider("候选组合数", options=[500,1000,3000,5000,10000], value=3000)
                use_constraints = st.checkbox("启用 A–E 单项及总含量约束", value=all(k in target_options for k in LIMITS))
                if st.button("搜索候选工艺参数", type="primary"):
                    rng = np.random.default_rng(2026)
                    bounds = artifact["feature_bounds"]
                    candidate_matrix = np.column_stack([
                        rng.uniform(bounds[f][0], bounds[f][1], int(n_candidates))
                        if bounds[f][1] > bounds[f][0] else np.repeat(bounds[f][0], int(n_candidates))
                        for f in artifact["feature_columns"]
                    ])
                    X = pd.DataFrame(candidate_matrix, columns=artifact["feature_columns"])
                    preds = {k: m.predict(X) for k,m in artifact["models"].items()}
                    result = X.copy()
                    for k,v in preds.items():
                        result[k] = v
                    feasible = np.ones(len(result), dtype=bool)
                    if use_constraints:
                        for k, limit in LIMITS.items():
                            if k not in result:
                                feasible &= False
                            else:
                                feasible &= result[k].to_numpy() >= limit
                        if "total_active_content_pct" in result:
                            total = result["total_active_content_pct"].to_numpy()
                        elif all(k in result for k in LIMITS):
                            total = result[list(LIMITS)].sum(axis=1).to_numpy()
                        else:
                            total = np.zeros(len(result))
                        feasible &= total >= 75.0
                    result["quality_constraint_pass"] = np.where(feasible, "通过", "未通过")
                    candidates = result[feasible] if use_constraints else result
                    if candidates.empty:
                        st.warning("在当前训练数据覆盖范围内未找到满足全部质量约束的候选方案。请增加实验数据或调整经验证的参数范围。")
                        candidates = result.sort_values(objective, ascending=(direction == "最小化")).head(20)
                    else:
                        candidates = candidates.sort_values(objective, ascending=(direction == "最小化"))
                        st.success(f"找到 {int(feasible.sum())} 组满足约束的候选方案。")
                    st.markdown("**排名靠前的候选方案**")
                    st.dataframe(candidates.head(20), use_container_width=True, hide_index=True)
                    st.download_button(
                        "下载优化候选结果 CSV",
                        data=candidates.to_csv(index=False).encode("utf-8-sig"),
                        file_name="本草智研_AI工艺优化候选结果.csv",
                        mime="text/csv",
                    )
                    st.warning("所有候选参数均须重新输入比赛软件计算验证。模型搜索结果不是已验证的生产工艺。")
            except Exception as exc:
                st.error(f"优化失败：{exc}")

    elif page == "管理员审核与训练":
        st.subheader("管理员控制台")
        admin = require_admin()
        pending = fetch_admin_submissions("pending")
        st.markdown(f"### 待审核提交（{len(pending)} 条）")
        if not pending:
            st.info("当前没有待审核数据。")
        for item in pending:
            flat = record_to_flat(item)
            with st.expander(f"{item.get('experiment_id')} · {item.get('submitter')} · {item.get('created_at')}"):
                st.write("来源：", item.get("source"))
                st.json({"inputs": item.get("inputs"), "outputs": item.get("outputs"), "notes": item.get("notes")})
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("审核通过", key=f"approve_{item['id']}", type="primary"):
                        set_submission_status(item["id"], "approved")
                        st.success("已加入共享训练数据。")
                        st.rerun()
                with c2:
                    if st.button("拒绝", key=f"reject_{item['id']}"):
                        set_submission_status(item["id"], "rejected", "管理员审核未通过")
                        st.warning("已拒绝该记录。")
                        st.rerun()
        st.divider()
        st.markdown("### 训练并发布共享模型")
        approved_df = build_training_frame(approved)
        st.write(f"当前已审核数据：**{len(approved_df)}** 条")
        available_targets = [
            t for t in TARGETS
            if t in approved_df.columns and approved_df[t].notna().sum() >= 6
        ]
        if not available_targets:
            st.warning("至少需要某个输出指标有6条有效记录，才可训练该指标。建议每个目标积累30–50组以上数据。")
        else:
            selected_targets = st.multiselect(
                "选择训练输出指标",
                options=available_targets,
                default=[t for t in available_targets if t in LIMITS or t == "total_active_content_pct"],
                format_func=lambda x: TARGET_LABELS.get(x,x),
            )
            c1,c2 = st.columns(2)
            with c1:
                n_estimators = st.slider("随机森林树数量", 100, 800, 300, 50)
            with c2:
                min_leaf = st.select_slider("叶节点最少样本数", options=[1,2,3,4,5], value=2)
            if st.button("训练并发布共享模型", type="primary", disabled=not selected_targets):
                with st.spinner("正在训练模型并发布到云端……"):
                    try:
                        artifact = train_models(approved_df, selected_targets, n_estimators, min_leaf)
                        version = publish_model(artifact)
                        st.success(f"模型 {version} 已发布。所有访问者现在都可以使用新模型。")
                        st.json(artifact["metrics"])
                        st.rerun()
                    except Exception as exc:
                        st.error(f"模型训练或发布失败：{exc}")

if __name__ == "__main__":
    main()
