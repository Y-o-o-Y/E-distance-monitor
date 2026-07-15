import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys


# ==================== 核心統計優化函數 ====================

def fast_sad(x):
    """計算所有兩兩絕對差之和 O(N log N)"""
    n = len(x)
    x_sorted = np.sort(x)
    return np.sum((2 * np.arange(1, n + 1) - n - 1) * x_sorted) * 2

def calculate_multisample_energy_optimized(data_matrix_np):
    """全球資產偏離度計算 (分母統一 n^2)"""
    n, k = data_matrix_np.shape
    sad_list = []
    within_means = []
    for i in range(k):
        sad_val = fast_sad(data_matrix_np[:, i])
        sad_list.append(sad_val)
        within_means.append(sad_val / (n**2))

    dist_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            combined = np.concatenate([data_matrix_np[:, i], data_matrix_np[:, j]])
            sum_dist_xy = fast_sad(combined) - sad_list[i] - sad_list[j]
            xy_mean = sum_dist_xy / (n**2)
            e_ij = 2 * xy_mean - within_means[i] - within_means[j]
            dist_matrix[i, j] = dist_matrix[j, i] = max(e_ij, 0)

    energy_stat = np.sum(dist_matrix) / 2
    deviations = np.sum(dist_matrix, axis=1) / (k - 1)
    return energy_stat, deviations

def optimized_energy_1d(x, y):
    """1D 能量距離計算 (分母統一 nx^2, ny^2)"""
    nx, ny = len(x), len(y)
    sad_x = fast_sad(x)
    sad_y = fast_sad(y)
    combined = np.concatenate([x, y])
    sad_combined = fast_sad(combined)
    sad_xy = (sad_combined - sad_x - sad_y) / 2
    
    term1 = 2 * (sad_xy / (nx * ny))
    term2 = sad_x / (nx**2)
    term3 = sad_y / (ny**2)
    return term1 - term2 - term3

# ==================== 1. 計算全球資產偏離度 ====================
print("--- 正在計算 [全球資產偏離度] ---")
global_tickers = ["^GSPC", "0050.TW", "^N225", "^KS11", "^NSEI", "^STOXX"]
# yf API 下載時，預設會回傳 Close (已調整除權息)
raw_global = yf.download(global_tickers, start="2003-01-01")['Close']
log_ret_global = np.log(raw_global.ffill() / raw_global.ffill().shift(1)).dropna()

window_size_global = 30
energy_values = []
dates_global = []
deviation_history = {ticker: [] for ticker in global_tickers}
log_returns_np = log_ret_global.values

for i in range(len(log_returns_np) - window_size_global + 1):
    window_data = log_returns_np[i : i + window_size_global, :]
    e_stat, devs = calculate_multisample_energy_optimized(window_data)
    energy_values.append(e_stat)
    for idx, ticker in enumerate(log_ret_global.columns):
        deviation_history[ticker].append(devs[idx])
    dates_global.append(log_ret_global.index[i + window_size_global - 1])

# 繪製全球圖表
fig_global = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
    subplot_titles=("Global Energy Distance", 
                    "Individual Asset Contribution")
)
fig_global.add_trace(go.Scatter(x=dates_global, y=energy_values, mode='lines', line=dict(color='#00CC96', width=2), name='Total Energy'), row=1, col=1)
for ticker in log_ret_global.columns:
    fig_global.add_trace(go.Scatter(x=dates_global, y=deviation_history[ticker], mode='lines', line=dict(width=1), name=ticker), row=2, col=1)
fig_global.update_layout(height=650, template='plotly_dark', hovermode='x unified', margin=dict(t=50, b=50, l=10, r=10))

# ==================== 2. 計算個股多尺度結構變化 ====================
print("--- 正在計算 [個股結構變化] ---")
# 如果執行命令時有傳入參數，就使用該參數當作 Ticker；否則預設為 'MU'
if len(sys.argv) > 1 and sys.argv[1].strip() != "":
    target_ticker = sys.argv[1].strip().upper()
else:
    target_ticker = 'MU'

ind_tickers = [target_ticker] # 動態決定目標

ed_configs = [(250, 125, 'Long (250/125)'), (120, 60, 'Mid (120/60)'), (60, 30, 'Short (60/30)')]
ER_window = 30

raw_ind = yf.download(ind_tickers, start="2023-01-01")['Close']
df_ind = raw_ind.to_frame() if isinstance(raw_ind, pd.Series) else raw_ind
log_ret_ind = np.log(df_ind / df_ind.shift(1)).dropna()

er_results = pd.DataFrame()
ed_multiscale = {}

for ticker in ind_tickers:
    rets = log_ret_ind[ticker].values
    for w_size, h_size, label in ed_configs:
        dists = []
        for i in range(len(rets) - w_size + 1):
            window_x = rets[i : i + h_size]
            window_y = rets[i + h_size : i + w_size]
            dists.append(optimized_energy_1d(window_x, window_y))
        ed_multiscale[f"{ticker}_{label}"] = pd.Series(dists, index=log_ret_ind.index[w_size-1:]) * 100

    net_change = (df_ind[ticker] - df_ind[ticker].shift(ER_window)).abs()
    volatility = df_ind[ticker].diff().abs().rolling(window=ER_window).sum()
    er_results[ticker] = (net_change / volatility).dropna()

# 繪製個股圖表
fig_ind = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
    subplot_titles=("Multi-scale Energy Distance", "Efficiency Ratio")
)
colors = {'Long (250/125)': '#636EFA', 'Mid (120/60)': '#AB63FA', 'Short (60/30)': '#00CC96'}
for ticker in ind_tickers:
    for _, _, label in ed_configs:
        key = f"{ticker}_{label}"
        fig_ind.add_trace(go.Scatter(x=ed_multiscale[key].index, y=ed_multiscale[key], name=f"ED {label}", line=dict(color=colors[label], width=1.5)), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=er_results.index, y=er_results[ticker], name=f"{ticker} ER", line=dict(color='#FFA15A', width=2)), row=2, col=1)
fig_ind.update_layout(height=650, template="plotly_dark", hovermode="x unified", margin=dict(t=50, b=50, l=10, r=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# ==================== 3. 生成美觀的雙頁籤 HTML ====================
print("--- 正在生成統一 Dashboard HTML ---")

# 使用純字串，避免 f-string 解析大括號衝突
html_template = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化統計量監控看板</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #111111;
            color: #f5f5f7;
        }
        .header {
            background-color: #1a1a1a;
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid #333;
        }
        .header h1 { margin: 0; font-size: 24px; color: #00CC96; }
        .header p { margin: 5px 0 0 0; font-size: 13px; color: #888; }
        
        /* 頁籤 Tab 樣式 */
        .tab-container {
            display: flex;
            justify-content: center;
            background-color: #1a1a1a;
            border-bottom: 1px solid #333;
        }
        .tab {
            padding: 15px 30px;
            cursor: pointer;
            border: none;
            background: none;
            color: #888;
            font-size: 16px;
            font-weight: bold;
            transition: 0.3s;
            outline: none;
        }
        .tab:hover { color: #fff; }
        .tab.active {
            color: #00CC96;
            border-bottom: 3px solid #00CC96;
        }
        
        .content-container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 0 15px;
        }
        .tab-content {
            display: none;
            background: #1e1e1e;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
        .tab-content.active { display: block; }

        /* 控制面板樣式 */
        .control-panel {
            background: #252525;
            border: 1px solid #444;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }
        .control-panel h4 { margin: 0 0 5px 0; width: 100%; color: #00CC96; }
        .input-group {
            display: flex;
            flex-direction: column;
        }
        .input-group label {
            font-size: 11px;
            color: #aaa;
            margin-bottom: 4px;
        }
        .control-panel input {
            background: #333;
            border: 1px solid #555;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            outline: none;
        }
        .control-panel input:focus {
            border-color: #00CC96;
        }
        .btn {
            background-color: #00CC96;
            color: #111;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            font-weight: bold;
            cursor: pointer;
            align-self: flex-end;
            transition: 0.2s;
        }
        .btn:hover { background-color: #00b383; }
        .btn:disabled { background-color: #555; color: #888; cursor: not-allowed; }
        #api-status { font-size: 12px; margin-top: 5px; width: 100%; }
    </style>
</head>
<body>

    <div class="header">
        <h1>📊 量化統計量監控看板</h1>
        <p>最後更新時間：{{LAST_UPDATE_TIME}} (UTC)</p>
    </div>

    <div class="tab-container">
        <button class="tab active" onclick="openTab(event, 'global-tab')">🌐 全球市場統計分佈</button>
        <button class="tab" onclick="openTab(event, 'individual-tab')">📈 個股統計分佈變化 ({{TARGET_TICKER}})</button>
    </div>

    <div class="content-container">
        <!-- 頁籤一：全球監控 -->
        <div id="global-tab" class="tab-content active">
            {{FIG_GLOBAL_HTML}}
        </div>

        <!-- 頁籤二：個股監控 -->
        <div id="individual-tab" class="tab-content">
            
            <!-- 網頁上手動更新控制面板 -->
            <div class="control-panel">
                <h4>🚀 網頁即時更換觀測個股</h4>
                <div class="input-group">
                    <label for="ticker-input">個股 Ticker (例如: AAPL, TSLA)</label>
                    <input type="text" id="ticker-input" placeholder="MU" value="{{TARGET_TICKER}}">
                </div>
                <div class="input-group" style="flex-grow: 1; max-width: 300px;">
                    <label for="token-input">GitHub Token (安全儲存於瀏覽器，僅需輸入一次)</label>
                    <input type="password" id="token-input" placeholder="ghp_xxxxxxxxxxxx">
                </div>
                <button class="btn" id="submit-btn" onclick="triggerGithubWorkflow()">送出更新</button>
                <div id="api-status"></div>
                <div style="font-size: 11px; color: #888; width: 100%;">
                    ※ 提示：第一次使用需要至 GitHub 申請一個經典的 Personal Access Token (PAT)，勾選 <code>workflow</code> 權限。<a href="https://github.com/settings/tokens/new" target="_blank" style="color: #00CC96;">點此快速申請</a>
                </div>
            </div>

            {{FIG_IND_HTML}}
        </div>
    </div>

    <script>
        // 初始化時從本地快取讀取 Token
        document.addEventListener("DOMContentLoaded", function() {
            const savedToken = localStorage.getItem("gh_token");
            if (savedToken) {
                document.getElementById("token-input").value = savedToken;
            }
        });

        function openTab(evt, tabId) {
            var i, tabcontent, tabs;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove("active");
            }
            tabs = document.getElementsByClassName("tab");
            for (i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove("active");
            }
            document.getElementById(tabId).classList.add("active");
            evt.currentTarget.classList.add("active");
            window.dispatchEvent(new Event('resize'));
        }

        // 發送 API 請求給 GitHub 來觸發 Workflow
        async function triggerGithubWorkflow() {
            const ticker = document.getElementById("ticker-input").value.trim().toUpperCase();
            const token = document.getElementById("token-input").value.trim();
            const statusDiv = document.getElementById("api-status");
            const btn = document.getElementById("submit-btn");

            if (!ticker) {
                alert("請輸入 Ticker 名稱！");
                return;
            }
            if (!token) {
                alert("請輸入 GitHub Personal Access Token (PAT) 以獲得發送權限！");
                return;
            }

            // 儲存 Token，下次打開網頁不用重新輸入
            localStorage.setItem("gh_token", token);
            
            btn.disabled = true;
            statusDiv.innerHTML = "⏳ 正在與 GitHub 通訊，發送更新指令...";
            statusDiv.style.color = "#888";

            const owner = "Y-o-o-Y"; 
            const repo = "E-distance-monitor";
            const workflowId = "run.yml"; // 對應 .github/workflows/run.yml

            const url = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflowId}/dispatches`;

            try {
                const response = await fetch(url, {
                    method: "POST",
                    headers: {
                        "Authorization": `Bearer ${token}`,
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28"
                    },
                    body: JSON.stringify({
                        ref: "main", // 使用 main 分支
                        inputs: {
                            ticker_input: ticker
                        }
                    })
                });

                if (response.status === 204) {
                    statusDiv.innerHTML = `✅ 成功發送指令！GitHub 已開始重新計算 <b>${ticker}</b> 的結構變化。<br>💡 請等待約 1 分鐘，然後「重新整理網頁」即可看到最新結果！`;
                    statusDiv.style.color = "#00CC96";
                } else {
                    const errorData = await response.json().catch(() => ({}));
                    statusDiv.innerHTML = `❌ 發送失敗 (代碼: ${response.status}): ${errorData.message || '請確認 Token 是否正確且具有 workflow 權限'}`;
                    statusDiv.style.color = "#FF4F56";
                }
            } catch (error) {
                statusDiv.innerHTML = `❌ 連線發生錯誤: ${error.message}`;
                statusDiv.style.color = "#FF4F56";
            } finally {
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

# 用標準的 .replace() 將 Python 變數精準填入，徹底避開大括號衝突！
html_content = html_template.replace("{{LAST_UPDATE_TIME}}", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
html_content = html_content.replace("{{TARGET_TICKER}}", target_ticker)
html_content = html_content.replace("{{FIG_GLOBAL_HTML}}", fig_global.to_html(full_html=False, include_plotlyjs='cdn'))
html_content = html_content.replace("{{FIG_IND_HTML}}", fig_ind.to_html(full_html=False, include_plotlyjs='cdn'))

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("Dashboard 生成成功！已寫入 index.html")
