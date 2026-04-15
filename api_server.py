from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image
import torch
import time
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from training.config import get_config
from training.inference import Inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EleGANt Makeup Transfer API")

# Trust Render's reverse proxy headers for HTTPS
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
    trusted_hosts=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = get_config()
model = Inference(config, type('Args', (), {'device': device})(), 'sow_pyramid_a5_e3d2_remapped.pth')

RESULTS_DIR = Path('api_results')
RESULTS_DIR.mkdir(exist_ok=True)

STYLES_DIR = Path('assets/images/makeup')

@app.get('/health')
@app.get('/api/health')
async def health():
    return {'status': 'ok', 'device': str(device)}

@app.get('/styles')
@app.get('/api/styles')
async def get_styles():
    try:
        styles = []
        style_files = sorted([f for f in STYLES_DIR.glob('make_styles_*.jpg')])

        for i, style_file in enumerate(style_files):
            styles.append({
                'id': f'style_{i+1}',
                'name': f'Makeup Style {i+1}',
                'thumbnail': f'/style/{i+1}'
            })

        logger.info(f"Returning {len(styles)} styles")
        return styles
    except Exception as e:
        logger.error(f"Error getting styles: {e}")
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get('/style/{style_id}')
@app.get('/api/style/{style_id}')
async def get_style_image(style_id: int):
    try:
        style_files = sorted([f for f in STYLES_DIR.glob('make_styles_*.jpg')])
        if 0 < style_id <= len(style_files):
            return FileResponse(style_files[style_id - 1], media_type='image/jpeg')
        return JSONResponse({'error': 'Style not found'}, status_code=404)
    except Exception as e:
        logger.error(f"Error getting style image: {e}")
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/transfer')
@app.post('/api/transfer')
async def transfer_makeup(image: UploadFile = File(...), style_id: str = Form(...), custom_style: UploadFile = None):
    try:
        logger.info(f"Transfer request: style_id={style_id}, custom_style={custom_style is not None}")

        source_img = Image.open(image.file).convert('RGB')

        if custom_style:
            style_img = Image.open(custom_style.file).convert('RGB')
            logger.info("Using custom style image")
        else:
            style_num = int(style_id.split('_')[-1]) if 'style_' in style_id else 1
            style_files = sorted([f for f in STYLES_DIR.glob('make_styles_*.jpg')])
            if 0 < style_num <= len(style_files):
                style_img = Image.open(style_files[style_num - 1]).convert('RGB')
                logger.info(f"Using preset style {style_num}")
            else:
                return JSONResponse({'error': 'Style not found'}, status_code=404)

        result = model.transfer(source_img, style_img, postprocess=True)

        if result is None:
            return JSONResponse({'error': 'Transfer failed'}, status_code=500)

        result_filename = f"result_{Path(image.filename).stem}_{int(time.time() * 1000)}.jpg"
        result_path = RESULTS_DIR / result_filename
        result.save(str(result_path), quality=95)

        logger.info(f"Result saved: {result_filename}")

        return {
    'status': 'success',
    'result_path': result_filename,
    'result_url': f'/api/result/{result_filename}'
        }
    except Exception as e:
        logger.error(f"Error in transfer: {e}")
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get('/result/{filename}')
@app.get('/api/result/{filename}')
async def get_result(filename: str):
    try:
        result_path = RESULTS_DIR / filename
        if result_path.exists():
            return FileResponse(result_path, media_type='image/jpeg')
        return JSONResponse({'error': 'Result not found'}, status_code=404)
    except Exception as e:
        logger.error(f"Error getting result: {e}")
        return JSONResponse({'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
