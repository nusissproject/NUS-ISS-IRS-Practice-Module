# Prompt
---

**Design a Gradio-based GUI for log case validation with the following specifications:**

**1. Log Case Selection**
- Provide a dropdown list populated from a text file (one log case per line).
- On selection, display the `image.jpg` from the selected log case folder in an image viewer, and render `label.json` in a JSON viewer.

**2. Validation Options**
- Provide three toggleable checkboxes: "JSON Schema", "Feature Outlier", and "SAM3 Validation".

**3. Validation Execution**
- A "Start Validation" button invokes a validation function that receives two arguments: the selected log case path and the list of enabled options.
- The function body should be a placeholder stub with a clear `# TODO: Add implementation` comment.
- The function writes its output to `data_validation_result.json` inside the selected log case folder.

**4. Results Display**
- After validation completes, parse `data_validation_result.json` and display three separate result tables, one for each category: "Labels with Issue", "False Positives", and "False Negatives".
- Each table has three columns: **ID**, **Issue Image** (rendered as an inline image), and **Data Failure Reason**.
- The `issue_image` field is a relative path within the log case folder; resolve and display the image inline in the table.

**Additional requirements:**
- Use `gr.Blocks` layout for flexible arrangement.
- Group the dropdown + image/JSON preview in one section, options + button in a second section, and the three result tables in a third section.
- Handle missing files gracefully (show a placeholder message if `image.jpg`, `label.json`, or `data_validation_result.json` is not found).

---

This tightens up ambiguities in the original — specifying the file format for the dropdown source, clarifying that `issue_image` is a path to resolve, calling for `gr.Blocks`, and adding error handling guidance. Want me to adjust the scope or detail level further?