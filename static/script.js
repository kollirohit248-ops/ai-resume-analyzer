document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("analyze-form");
  const button = document.getElementById("analyze-button");
  const statusEl = document.getElementById("status");
  const resultsEl = document.getElementById("results");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    hideStatus();
    resultsEl.hidden = true;

    const formData = new FormData(form);

    setLoading(true);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        showStatus(data.error || "Something went wrong. Please try again.", "error");
        return;
      }

      renderResults(data);
    } catch (err) {
      showStatus("Could not reach the server. Please try again.", "error");
    } finally {
      setLoading(false);
    }
  });

  function setLoading(isLoading) {
    button.disabled = isLoading;
    button.textContent = isLoading ? "Analyzing..." : "Analyze";
    if (isLoading) {
      showStatus("Analyzing your resume against the job description...", "info");
    }
  }

  function showStatus(message, kind) {
    statusEl.textContent = message;
    statusEl.className = `status ${kind}`;
    statusEl.hidden = false;
  }

  function hideStatus() {
    statusEl.hidden = true;
  }

  function renderResults(data) {
    hideStatus();
    resultsEl.hidden = false;

    const score = data.score;
    document.getElementById("score-number").textContent = score.final_score;
    document.getElementById("low-confidence-note").hidden = !score.low_confidence;

    setComponent("comp-required", score.required_skill_score);
    setComponent("comp-general", score.general_keyword_score);
    setComponent("comp-experience", score.experience_project_score);
    setComponent("comp-education", score.education_score);

    fillList("matched-required", score.matched_required_skills);
    fillList("missing-required", score.missing_required_skills);
    fillList("matched-general", score.matched_general_skills);
    fillList("missing-general", score.missing_general_skills);

    const analysisError = document.getElementById("analysis-error");
    const analysisContent = document.getElementById("analysis-content");

    if (data.analysis) {
      analysisError.hidden = true;
      analysisContent.hidden = false;
      fillList("strengths-list", data.analysis.strengths);
      fillList("weaknesses-list", data.analysis.weaknesses);
      fillList("suggestions-list", data.analysis.improvement_suggestions);
      document.getElementById("summary-text").textContent = data.analysis.compatibility_summary;
    } else {
      analysisContent.hidden = true;
      analysisError.hidden = false;
      analysisError.textContent = data.analysis_error || "AI analysis is unavailable.";
    }
  }

  function setComponent(elementId, value) {
    document.getElementById(elementId).textContent = value === null || value === undefined ? "N/A" : `${value}%`;
  }

  function fillList(elementId, items) {
    const list = document.getElementById(elementId);
    list.innerHTML = "";
    (items || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      list.appendChild(li);
    });
  }
});
