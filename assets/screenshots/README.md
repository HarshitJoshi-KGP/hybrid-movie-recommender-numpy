# Screenshots

This folder is a placeholder. Before the public push, add real screenshots
here (I can't generate genuine app screenshots myself since that requires
actually running Streamlit/FastAPI with a live browser) and then reference
them from the main `README.md`:

| File to add | Shows | Suggested command |
|---|---|---|
| `streamlit-ui.png` | Full Streamlit demo page (sidebar + both columns) | `streamlit run app.py`, then screenshot the browser tab |
| `recommendation-output.png` | Close-up of the "Top Recommendations" table for a populated user, ideally with `explain` toggled on | Same as above, after clicking "Get Recommendations" |
| `cold-start-demo.png` | The same, with "Cold-Start Mode" checked, to show the fallback path | Same as above |
| `fastapi-swagger.png` | `/docs` Swagger UI showing `/recommend/{user_id}`, `/recommend/batch`, `/similar/{movie_id}`, and `/health` | `uvicorn src.api:app --reload --port 8000`, open `http://127.0.0.1:8000/docs` |

The evaluation-metrics chart no longer needs a manual screenshot --
`notebooks/evaluation_metrics.png` is generated automatically and already
linked from the root README's "Example Outputs" section.

Keep each image under ~500KB (PNG, cropped to the relevant region) so the
repo stays lightweight. Once added, replace the placeholder image links in
the root `README.md`'s "Example Outputs" section with these paths.
