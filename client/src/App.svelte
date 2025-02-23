<script>
    import "./app.css";
    import { marked } from "marked";
    let uploading = false;
    let result = null;
    let error = null;
    let dragOver = false;

    const API_BASE = "http://localhost:5000";

    marked.setOptions({
        breaks: true,
        headerIds: false,
    });

    function renderMarkdown(text) {
        try {
            return marked(text);
        } catch (e) {
            console.error("Error rendering markdown:", e);
            return text.replace(/\n/g, "<br>");
        }
    }

    function handleDragOver(e) {
        e.preventDefault();
        dragOver = true;
    }

    function handleDragLeave() {
        dragOver = false;
    }

    function handleDrop(e) {
        e.preventDefault();
        dragOver = false;
        const file = e.dataTransfer?.files[0];
        if (file) handleFile(file);
    }

    function handleFileInput(event) {
        const file = event.target.files[0];
        if (file) handleFile(file);
    }

    async function handleFile(file) {
        uploading = true;
        error = null;
        result = null;

        const formData = new FormData();
        formData.append("video", file);

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            if (response.ok) {
                result = {
                    ...data,
                    first_frame_path: `${API_BASE}${data.first_frame_path}`,
                    final_map_path: `${API_BASE}${data.final_map_path}`,
                };
            } else {
                error = data.error;
            }
        } catch (e) {
            error = "Upload failed: " + e.message;
        } finally {
            uploading = false;
        }
    }
</script>

<main>
    <div class="container">
        <h1>Play Analysis</h1>

        <div
            class="upload-zone"
            class:drag-over={dragOver}
            on:dragover={handleDragOver}
            on:dragleave={handleDragLeave}
            on:drop={handleDrop}
        >
            <input type="file" accept=".mp4" on:change={handleFileInput} disabled={uploading} id="file-input" />
            <label for="file-input">
                <div class="upload-content">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    >
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                    <span>{uploading ? "Processing..." : "Drop video or click to upload"}</span>
                </div>
            </label>
        </div>

        {#if error}
            <div class="alert error">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                >
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <span>{error}</span>
            </div>
        {/if}

        {#if result}
            <div class="result-container">
                <div class="images-grid">
                    <div class="image-card">
                        <h3>Starting Formation</h3>
                        <img src={result.first_frame_path} alt="Starting formation" loading="lazy" />
                    </div>
                    <div class="image-card">
                        <h3>Movement Analysis</h3>
                        <img src={result.final_map_path} alt="Player tracking map" loading="lazy" />
                    </div>
                </div>

                <div class="analysis-card">
                    <h2>AI Analysis</h2>
                    {#if result.analysis}
                        <div class="analysis-content markdown">
                            {@html renderMarkdown(result.analysis)}
                        </div>
                    {:else}
                        <p class="no-analysis">Analysis not available</p>
                    {/if}
                </div>
            </div>
        {/if}
    </div>
</main>

<style>
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 2rem;
        text-align: center;
    }

    .upload-zone {
        background: #ffffff;
        border: 2px dashed #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }

    .upload-zone:hover,
    .drag-over {
        border-color: #2563eb;
        background: #f8fafc;
    }

    .upload-zone input {
        display: none;
    }

    .upload-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
        color: #64748b;
    }

    .upload-content svg {
        width: 48px;
        height: 48px;
        color: #2563eb;
    }

    .alert {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .error {
        background: #fee2e2;
        color: #dc2626;
    }

    .result-container {
        display: flex;
        flex-direction: column;
        gap: 2rem;
    }

    .images-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
    }

    .image-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    .image-card h3 {
        padding: 1rem;
        margin: 0;
        background: #f8fafc;
        font-size: 1.25rem;
        color: #1e293b;
    }

    .image-card img {
        width: 100%;
        height: auto;
        display: block;
    }

    .analysis-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    .analysis-card h2 {
        margin-top: 0;
        color: #1e293b;
        font-size: 1.5rem;
    }

    .analysis-content {
        color: #475569;
        line-height: 1.7;
    }

    .no-analysis {
        color: #94a3b8;
        text-align: center;
        padding: 2rem;
    }

    .markdown {
        color: #1e293b;
    }

    .markdown :global(h1) {
        font-size: 1.8em;
        margin: 1em 0 0.5em;
    }

    .markdown :global(h2) {
        font-size: 1.5em;
        margin: 1em 0 0.5em;
    }

    .markdown :global(h3) {
        font-size: 1.2em;
        margin: 1em 0 0.5em;
    }

    .markdown :global(p) {
        margin: 0.8em 0;
    }

    .markdown :global(ul),
    .markdown :global(ol) {
        margin: 0.8em 0;
        padding-left: 2em;
    }

    .markdown :global(li) {
        margin: 0.3em 0;
    }

    .markdown :global(code) {
        background: #f1f5f9;
        padding: 0.2em 0.4em;
        border-radius: 4px;
        font-size: 0.9em;
    }

    .markdown :global(pre) {
        background: #f1f5f9;
        padding: 1em;
        border-radius: 8px;
        overflow-x: auto;
    }

    .markdown :global(blockquote) {
        border-left: 4px solid #e2e8f0;
        margin: 1em 0;
        padding-left: 1em;
        color: #64748b;
    }

    .markdown :global(hr) {
        border: none;
        border-top: 2px solid #e2e8f0;
        margin: 2em 0;
    }

    .markdown :global(a) {
        color: #2563eb;
        text-decoration: none;
    }

    .markdown :global(a:hover) {
        text-decoration: underline;
    }

    @media (max-width: 768px) {
        .container {
            padding: 1rem;
        }

        h1 {
            font-size: 2rem;
        }

        .images-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
