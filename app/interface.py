import os
import gradio as gr
from agent.rag_agent import MistralRAGAgent

# Global agent placeholder, initialized when the app starts
agent = None

def get_agent():
    global agent
    if agent is None:
        agent = MistralRAGAgent()
    return agent

def qa_predict(question: str):
    """
    Function invoked by Gradio. Resolves the question via the RAG agent
    and formats the outputs.
    """
    if not question.strip():
        return "Veuillez poser une question valide.", "Pas de sources."
        
    rag = get_agent()
    result = rag.query(question)
    
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    
    # Format sources as markdown list with links
    if not sources:
        sources_md = "*Aucune source trouvée dans la base locale.*"
    else:
        sources_md = "### Sources consultées :\n"
        for src in sources:
            title = src.get("title", "Lien")
            url = src.get("url", "#")
            src_type = "GitHub" if src.get("type") == "github" else "Documentation"
            sources_md += f"- **[{title}]({url})** ({src_type})\n"
            
    return answer, sources_md

def launch_app(share=False):
    # Load stylesheet to look premium (dark base, mistral-themed colors like orange/deep-grey)
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    #title-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(135deg, #ff5e3a 0%, #ff2a6d 100%);
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(255, 42, 109, 0.2);
    }
    #title-container h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }
    #title-container p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .orange-btn {
        background-color: #ff5e3a !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .orange-btn:hover {
        background-color: #ff2a6d !important;
    }
    .gr-input, .gr-box {
        border-color: #1f2937 !important;
        background-color: #111827 !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Mistral AI Q&A Agent RAG") as demo:
        with gr.Column(elem_id="title-container"):
            gr.HTML("<h1>Mistral AI Q&A RAG Agent</h1><p>Proof of Concept - Phase 1</p>")
            
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Posez votre question sur l'écosystème Mistral AI :",
                    placeholder="Ex: Comment faire du fine-tuning avec mistral-finetune ?",
                    lines=3,
                    elem_classes="gr-input"
                )
                submit_btn = gr.Button("Rechercher & Répondre", elem_classes="orange-btn")
                
            with gr.Column(scale=3):
                answer_output = gr.Markdown(
                    label="Réponse de l'agent",
                    value="*La réponse s'affichera ici après validation de votre question.*"
                )
                sources_output = gr.Markdown(
                    label="Sources",
                    value="*Les sources consultées s'afficheront ici.*"
                )
                
        # Add examples to make it easy to start
        gr.Examples(
            examples=[
                ["Quels sont les dépôts principaux de Mistral AI ?"],
                ["Comment utiliser le SDK python de Mistral ?"],
                ["Qu'est-ce que mistral-common et à quoi ça sert ?"],
                ["Explique comment fine-tuner un modèle Mistral."]
            ],
            inputs=question_input,
            label="Exemples de questions"
        )
        
        submit_btn.click(
            fn=qa_predict,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
        # Also submit on hitting enter
        question_input.submit(
            fn=qa_predict,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
    demo.launch(server_name="127.0.0.1", server_port=7860, share=share)

if __name__ == "__main__":
    launch_app()
