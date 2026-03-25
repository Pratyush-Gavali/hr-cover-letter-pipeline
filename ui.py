"""
Gradio UI for the HR Cover Letter Intelligence Pipeline.

Four tabs:
  1. Job Setup      — seed a job description for a job_id
  2. Upload         — upload cover letters and view live SVA scores
  3. Applicants     — browse all applicants for a job with score breakdown
  4. HR Query       — natural language talent search via the RAG pipeline
"""

from __future__ import annotations
import os
import requests
import gradio as gr

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api/v1")
HEADERS = {"X-User-ID": "hr_ui_user"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _post(path: str, **kwargs) -> dict:
    r = requests.post(f"{API_BASE}{path}", headers=HEADERS, timeout=120, **kwargs)
    r.raise_for_status()
    return r.json()

def _get(path: str, **kwargs) -> dict:
    r = requests.get(f"{API_BASE}{path}", headers=HEADERS, timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


# ── Tab 1: Job Setup ───────────────────────────────────────────────────────────

def seed_jd(job_id: str, jd_text: str) -> str:
    if not job_id.strip():
        return "⚠️ Please enter a Job ID."
    if not jd_text.strip():
        return "⚠️ Please paste the job description."
    try:
        data = _post(f"/jd/{job_id.strip()}", json={"jd_text": jd_text.strip()})
        return f"✅ Job description seeded for **{data['job_id']}** ({data['length']} characters)."
    except Exception as e:
        return f"❌ Error: {e}"


# ── Tab 2: Upload ──────────────────────────────────────────────────────────────

def upload_cover_letter(job_id: str, file) -> tuple[str, list]:
    if not job_id.strip():
        return "⚠️ Please enter a Job ID.", []
    if file is None:
        return "⚠️ Please upload a file.", []

    mime_map = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt":  "text/plain",
    }
    ext = os.path.splitext(file.name)[1].lower()
    mime = mime_map.get(ext, "application/octet-stream")

    try:
        with open(file.name, "rb") as f:
            data = _post(
                f"/covers/{job_id.strip()}",
                files={"file": (os.path.basename(file.name), f, mime)},
            )

        summary = (
            f"✅ **Uploaded successfully**\n\n"
            f"- Applicant ID: `{data['applicant_id']}`\n"
            f"- Chunks extracted: **{data['chunk_count']}**\n"
            f"- PII entities masked: **{data['entity_count']}**\n"
            f"- JD match score: **{data['jd_match_score']:.3f}**\n"
            f"- AI probability: **{data['ai_probability']:.3f}** "
            f"({'⚠️ Possible AI-written' if data['ai_probability'] > 0.5 else '✅ Likely human-written'})\n"
        )

        scores_table = [[
            data["applicant_id"],
            f"{data['jd_match_score']:.3f}",
            f"{data['ai_probability']:.3f}",
            f"{1 - data['ai_probability']:.3f}",
            str(data["chunk_count"]),
            str(data["entity_count"]),
        ]]
        return summary, scores_table

    except Exception as e:
        return f"❌ Error: {e}", []


# ── Tab 3: Applicants ──────────────────────────────────────────────────────────

def list_applicants(job_id: str) -> tuple[list, str]:
    if not job_id.strip():
        return [], "⚠️ Please enter a Job ID."
    try:
        data = _get(f"/applicants/{job_id.strip()}")
        applicants = data.get("applicants", [])
        if not applicants:
            return [], f"No applicants found for job **{job_id}**."

        rows = []
        for a in applicants:
            ai_p = a.get("ai_probability") or 0.0
            jd_m = a.get("jd_match_score") or 0.0
            flag = "⚠️ AI?" if ai_p > 0.5 else "✅ Human"
            rows.append([
                a["applicant_id"],
                f"{jd_m:.3f}",
                f"{ai_p:.3f}",
                f"{1 - ai_p:.3f}",
                flag,
                a.get("blob_path", ""),
            ])

        return rows, f"Found **{len(applicants)}** applicant(s) for job `{job_id}`."
    except Exception as e:
        return [], f"❌ Error: {e}"


# ── Tab 4: HR Query ────────────────────────────────────────────────────────────

def run_query(
    job_id: str,
    prompt: str,
    ai_prob_max: float,
    min_match_score: float,
) -> tuple[str, list]:
    if not job_id.strip():
        return "⚠️ Please enter a Job ID.", []
    if not prompt.strip():
        return "⚠️ Please enter a query.", []
    try:
        data = _post("/query", json={
            "prompt": prompt.strip(),
            "job_id": job_id.strip(),
            "ai_prob_max": ai_prob_max,
            "min_match_score": min_match_score,
        })

        response_md = f"### Response\n\n{data['response']}"

        candidates = data.get("top_candidates", [])
        rows = []
        for c in candidates:
            ai_p = c.get("ai_probability") or 0.0
            rows.append([
                c.get("applicant_id", ""),
                f"{c.get('jd_match_score', 0):.3f}",
                f"{ai_p:.3f}",
                f"{c.get('rerank_score', 0):.3f}",
                "⚠️ AI?" if ai_p > 0.5 else "✅ Human",
                (c.get("chunk_text") or "")[:120] + "…",
            ])

        return response_md, rows
    except Exception as e:
        return f"❌ Error: {e}", []


# ── Layout ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="HR Cover Letter Intelligence Pipeline") as demo:

    gr.Markdown(
        """
        # 🧠 HR Cover Letter Intelligence Pipeline
        Semantic Velocity Analysis · BGE-M3 Hybrid Search · PII Masking · AI Authorship Detection
        """
    )

    with gr.Tabs():

        # ── Tab 1: Job Setup ───────────────────────────────────────────────────
        with gr.Tab("📋 Job Setup"):
            gr.Markdown("Seed a job description so the SVA engine can score cover letters against it.")
            with gr.Row():
                jd_job_id = gr.Textbox(label="Job ID", placeholder="e.g. job_ba_pnbank", scale=1)
            jd_text = gr.Textbox(
                label="Job Description",
                placeholder="Paste the full job description here…",
                lines=15,
            )
            jd_btn = gr.Button("Seed Job Description", variant="primary")
            jd_status = gr.Markdown()

            jd_btn.click(seed_jd, inputs=[jd_job_id, jd_text], outputs=jd_status)

        # ── Tab 2: Upload ──────────────────────────────────────────────────────
        with gr.Tab("📤 Upload Cover Letter"):
            gr.Markdown("Upload a PDF, DOCX, or TXT cover letter. SVA scores are computed instantly.")
            with gr.Row():
                upload_job_id = gr.Textbox(label="Job ID", placeholder="e.g. job_ba_pnbank", scale=1)
            upload_file = gr.File(
                label="Cover Letter (PDF / DOCX / TXT)",
                file_types=[".pdf", ".docx", ".txt"],
            )
            upload_btn = gr.Button("Upload & Analyse", variant="primary")
            upload_status = gr.Markdown()
            upload_table = gr.Dataframe(
                headers=["Applicant ID", "JD Match", "P(AI)", "Human Conf.", "Chunks", "PII Entities"],
                datatype=["str", "str", "str", "str", "str", "str"],
                label="SVA Scores",
                interactive=False,
            )

            upload_btn.click(
                upload_cover_letter,
                inputs=[upload_job_id, upload_file],
                outputs=[upload_status, upload_table],
            )

        # ── Tab 3: Applicants ──────────────────────────────────────────────────
        with gr.Tab("👥 Applicant Dashboard"):
            gr.Markdown("Browse all indexed applicants for a job, sorted by JD match score.")
            with gr.Row():
                appl_job_id = gr.Textbox(label="Job ID", placeholder="e.g. job_ba_pnbank", scale=2)
                appl_btn = gr.Button("Load Applicants", variant="primary", scale=1)
            appl_status = gr.Markdown()
            appl_table = gr.Dataframe(
                headers=["Applicant ID", "JD Match", "P(AI)", "Human Conf.", "Auth. Flag", "Blob Path"],
                datatype=["str", "str", "str", "str", "str", "str"],
                label="Applicants",
                interactive=False,
            )

            appl_btn.click(
                list_applicants,
                inputs=[appl_job_id],
                outputs=[appl_table, appl_status],
            )

        # ── Tab 4: HR Query ────────────────────────────────────────────────────
        with gr.Tab("🔍 HR Talent Query"):
            gr.Markdown(
                "Ask a natural language question. The RAG pipeline retrieves, reranks, "
                "and synthesises a grounded answer from masked cover letter excerpts."
            )
            with gr.Row():
                query_job_id = gr.Textbox(label="Job ID", placeholder="e.g. job_ba_pnbank", scale=1)
            query_prompt = gr.Textbox(
                label="Query",
                placeholder="e.g. Who has the strongest Agile and compliance experience?",
                lines=3,
            )
            with gr.Row():
                ai_prob_slider = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.05,
                    label="Max AI Probability (lower = prefer human-written)",
                )
                match_slider = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.05,
                    label="Min JD Match Score",
                )
            query_btn = gr.Button("Run Query", variant="primary")
            query_response = gr.Markdown(label="LLM Response")
            query_table = gr.Dataframe(
                headers=["Applicant ID", "JD Match", "P(AI)", "Rerank Score", "Auth. Flag", "Top Excerpt"],
                datatype=["str", "str", "str", "str", "str", "str"],
                label="Top Candidates",
                interactive=False,
                wrap=True,
            )

            query_btn.click(
                run_query,
                inputs=[query_job_id, query_prompt, ai_prob_slider, match_slider],
                outputs=[query_response, query_table],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
