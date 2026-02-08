import gradio as gr
from dotenv import load_dotenv
from scripts.answer import answer_question

load_dotenv(override=True)

def format_sources(chunks):
    if not chunks:
        return ""
    text = "### 📚 Legal Sources Used\n"
    unique_sources = {c['meta'].get('source', 'Unknown') for c in chunks}
    for source in unique_sources:
        text += f"- {source}\n"
    return text

def chat(user_message, history):
    # 1. Initialize history if None
    if history is None:
        history = []
        
    if not isinstance(user_message, str) or not user_message.strip():
        return "", history, ""

    # --- THE FIX: Pass 'history' to the answer function ---
    # We pass the raw Gradio history (list of tuples) to our backend
    answer, chunks = answer_question(user_message, history)

    # 3. Append to history as DICTIONARIES for Gradio 5.x compatibility
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    
    sources_text = format_sources(chunks)

    return "", history, sources_text

def main():
    with gr.Blocks(title="India Travel Law Assistant") as app:
        gr.Markdown("# 🇮🇳 India Travel Law Assistant")
        gr.Markdown("Ask questions about Indian laws relevant to travellers.")

        with gr.Row():
            # Standard Gradio Chatbot
            chatbot = gr.Chatbot(height=500) 
            sources_display = gr.Markdown()

        msg = gr.Textbox(
            placeholder="e.g. Is public drinking legal in India?",
            show_label=False
        )

        msg.submit(
            chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, sources_display],
            concurrency_limit=10
        )

    app.launch(inbrowser=True)

if __name__ == "__main__":
    main()