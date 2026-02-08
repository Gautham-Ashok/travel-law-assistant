import chainlit as cl
from scripts.answer import answer_question

@cl.on_chat_start
async def start():
    """
    Initializes the chat session.
    """
    # Initialize session history
    cl.user_session.set("history", [])
    
    # Send a welcome message
    await cl.Message(
        content="🇮🇳 **Namaste! I am your India Travel Law Assistant.**\n\nI can help you with:\n- 🍺 **Alcohol Laws** (Dry states, legal age)\n- 👮 **Police Interactions** (Arrest rights, 112 helpline)\n- 💊 **Drug Laws** (NDPS Act penalties)\n- 🛂 **Visa Rules** (Overstaying, FRRO)\n\n*Ask me a question to get started!*",
        author="Legal Assistant"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Main chat loop: Receives user message, calls RAG backend, sends response.
    """
    # 1. Get previous history
    history = cl.user_session.get("history")
    
    # 2. Show a "Thinking..." loader (Chainlit UI feature)
    msg = cl.Message(content="")
    await msg.send()
    
    # 3. Call the Python Backend (Wrap synchronous function to be async)
    # We pass the user's message content and the chat history
    final_answer, chunks = await cl.make_async(answer_question)(message.content, history)

    # 4. Format the Legal Sources (Citations)
    # This creates clickable "Elements" in the Chainlit UI
    source_elements = []
    if chunks:
        for i, chunk in enumerate(chunks):
            source_name = chunk['meta'].get('source', 'Unknown Law')
            # detailed text for the side panel
            source_content = f"**Source:** {source_name}\n\n{chunk['text']}"
            
            source_elements.append(
                cl.Text(
                    name=f"Legal Source {i+1}", 
                    content=source_content, 
                    display="side"  # Displays in the collapsible side panel
                )
            )

    # 5. Update History (Append new turn)
    # Note: Chainlit manages the UI history, but we keep our own list for the RAG context
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": final_answer})
    cl.user_session.set("history", history)

    # 6. Send Final Response
    msg.content = final_answer
    msg.elements = source_elements
    await msg.update()