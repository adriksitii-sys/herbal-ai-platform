# 本草智研 V2：中药提取纯化工艺云端 AI 优化平台

本项目使用 Streamlit + Supabase。公开访客可提交实验数据；新数据默认进入待审核队列；管理员审核通过后，数据才进入共享训练集。管理员可训练并发布共享随机森林模型，所有访问者可使用已发布模型进行质量预测和工艺候选搜索。

## 部署文件
- app.py：Streamlit 应用入口
- requirements.txt：Python 依赖
- supabase_schema.sql：Supabase 数据库初始化脚本
- .streamlit/config.toml：Streamlit 页面配置
- secrets.example.toml：Secrets 配置示例（仅占位符，不含真实密钥）

## 部署顺序
1. 在 Supabase SQL Editor 执行 supabase_schema.sql。
2. 在 Streamlit Community Cloud 连接本仓库，Main file path 设置为 app.py。
3. 在 Streamlit App Settings → Secrets 中配置 SUPABASE_URL、SUPABASE_ANON_KEY、SUPABASE_SERVICE_ROLE_KEY 和 ADMIN_PASSWORD。
4. 保存并部署。不要将真实密钥提交到 GitHub。

## 注意
公开提交的数据会进入共享数据库；只有管理员审核通过的数据用于模型训练。模型预测和优化结果仅用于筛选候选方案，必须通过比赛软件或实验进一步验证。本系统不是经验证的 GMP 生产控制系统。