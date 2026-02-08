import gradio as gr
import pandas as pd
from collections import defaultdict
from scripts.eval import run_all_retrieval_tests, run_all_answer_tests

# --- UI LOGIC ---

def run_retrieval_ui(progress=gr.Progress()):
    results_list = []
    category_mrr = defaultdict(list)
    
    count = 0
    total_mrr = 0
    total_ndcg = 0
    total_cov = 0

    for test, res, prog in run_all_retrieval_tests():
        count += 1
        total_mrr += res.mrr
        total_ndcg += res.ndcg
        total_cov += res.keyword_coverage
        
        category_mrr[test.category].append(res.mrr)
        progress(prog, desc=f"Testing: {test.question[:30]}...")

    # Summary Stats
    avg_mrr = total_mrr / count if count else 0
    avg_ndcg = total_ndcg / count if count else 0
    avg_cov = total_cov / count if count else 0

    # HTML Report - FIXED TEXT COLORS
    html = f"""
    <div style="display: flex; gap: 20px;">
        <div style="background: #e6fffa; padding: 20px; border-radius: 10px; flex: 1;">
            <h3 style="color: #444; margin-bottom: 5px;">MRR</h3>
            <h1 style="color: #00a884; margin-top: 0;">{avg_mrr:.3f}</h1>
        </div>
        <div style="background: #e6fffa; padding: 20px; border-radius: 10px; flex: 1;">
            <h3 style="color: #444; margin-bottom: 5px;">nDCG</h3>
            <h1 style="color: #00a884; margin-top: 0;">{avg_ndcg:.3f}</h1>
        </div>
        <div style="background: #e6fffa; padding: 20px; border-radius: 10px; flex: 1;">
            <h3 style="color: #444; margin-bottom: 5px;">Coverage</h3>
            <h1 style="color: #00a884; margin-top: 0;">{avg_cov:.1f}%</h1>
        </div>
    </div>
    """

    # Chart Data
    chart_data = []
    for cat, scores in category_mrr.items():
        chart_data.append({"Category": cat, "MRR": sum(scores)/len(scores)})
    
    return html, pd.DataFrame(chart_data)

def run_answer_ui(progress=gr.Progress()):
    category_acc = defaultdict(list)
    total_acc = 0
    total_comp = 0
    total_rel = 0
    count = 0

    for test, res, prog in run_all_answer_tests():
        count += 1
        total_acc += res.accuracy
        total_comp += res.completeness
        total_rel += res.relevance
        category_acc[test.category].append(res.accuracy)
        progress(prog, desc=f"Judging: {test.question[:30]}...")

    avg_acc = total_acc / count if count else 0
    avg_comp = total_comp / count if count else 0
    avg_rel = total_rel / count if count else 0

    # HTML Report - FIXED TEXT COLORS
    html = f"""
    <div style="display: flex; gap: 20px;">
        <div style="background: #fff5f5; padding: 20px; border-radius: 10px; flex: 1;">
            <h3 style="color: #444; margin-bottom: 5px;">Accuracy</h3>
            <h1 style="color: #c53030; margin-top: 0;">{avg_acc:.2f}/5</h1>
        </div>
        <div style="background: #fff5f5; padding: 20px; border-radius: 10px; flex: 1;">
            <h3 style="color: #444; margin-bottom: 5px;">Completeness</h3>
            <h1 style="color: #c53030; margin-top: 0;">{avg_comp:.2f}/5</h1>
        </div>
        <div style="background: #fff5f5; padding: 20px; border-radius: 10px; flex: 1;">
            <h3 style="color: #444; margin-bottom: 5px;">Relevance</h3>
            <h1 style="color: #c53030; margin-top: 0;">{avg_rel:.2f}/5</h1>
        </div>
    </div>
    """
    
    chart_data = []
    for cat, scores in category_acc.items():
        chart_data.append({"Category": cat, "Accuracy": sum(scores)/len(scores)})

    return html, pd.DataFrame(chart_data)

# --- APP LAYOUT ---

with gr.Blocks(title="🇮🇳 Legal RAG Eval", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚖️ India Travel Law - System Evaluation")
    
    with gr.Tab("Retrieval Metrics"):
        gr.Markdown("Measure how well we find the correct keywords in the vector DB.")
        btn_ret = gr.Button("Run Retrieval Test", variant="primary")
        out_html_ret = gr.HTML()
        out_plot_ret = gr.BarPlot(x="Category", y="MRR", title="MRR by Category", y_lim=[0,1])
        btn_ret.click(run_retrieval_ui, outputs=[out_html_ret, out_plot_ret])

    with gr.Tab("LLM Judge Metrics"):
        gr.Markdown("Use GPT-4o to grade the accuracy of generated answers.")
        btn_ans = gr.Button("Run LLM Judge", variant="primary")
        out_html_ans = gr.HTML()
        out_plot_ans = gr.BarPlot(x="Category", y="Accuracy", title="Accuracy by Category", y_lim=[0,5])
        btn_ans.click(run_answer_ui, outputs=[out_html_ans, out_plot_ans])

if __name__ == "__main__":
    app.launch()