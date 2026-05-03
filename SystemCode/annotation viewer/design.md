
> I have a JSON annotation file containing object polygon masks (COCO-style or similar). Build a **standalone web application** using plain HTML, CSS, and JavaScript — structured as a scaffolded project (e.g. `index.html`, `app.js`, `style.css`) — that supports both **viewing** and **editing** polygon mask annotations on images.
>
> **Viewer (left panel):**
> - Load and display an image with all polygon masks overlaid as semi-transparent filled shapes with visible outlines
> - Support **zoom in / zoom out** (mouse wheel or buttons), with the mask layer scaling in sync with the image
> - Allow the user to **select a mask** either by clicking directly on it in the image view or by clicking its entry in the polygon list panel
> - Highlight the selected mask visually (e.g. distinct border or fill colour)
>
> **Polygon list panel (right panel):**
> - Display all annotations as a scrollable list showing each mask's name and a colour swatch
> - Clicking a list item selects the corresponding mask in the image view and vice versa
>
> **Editor:**
> - Allow the user to draw new masks using one of four tools: **polygon** (click-to-place vertices, double-click to close), **circle**, **ellipse**, and **rectangle**
> - After completing a shape, show a form/modal where the user can enter a **mask name** and any number of **custom key-value attributes**
> - Allow the user to **delete** or **re-edit** existing masks
>
> **Export:**
> - Provide an **Export JSON** button that downloads the updated annotation file (preserving the original structure and appending any new or edited masks)
>
> **Constraints:**
> - No frameworks or build tools — pure HTML/CSS/JS only
> - All files must be self-contained and openable by double-clicking `index.html` (no local server required)
> - Use a `<canvas>` element for rendering the image and mask overlays
> - Structure the project as separate files: `index.html`, `style.css`, `app.js` (and any logical sub-modules if needed)
>
> Produce a working first draft for review. Do not implement all edge cases yet — focus on the core view/edit/export loop being functional end to end.

