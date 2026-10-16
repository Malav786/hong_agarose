"""
NanoStack 3D Hugging Face Space Entrypoint.
Runs the FastAPI web dashboard on port 7860 for public demonstration.
"""

import os
import uvicorn
from final_system.main import app

if __name__ == "__main__":
    # Hugging Face Spaces forwards port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
