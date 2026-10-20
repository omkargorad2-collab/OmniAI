import os, io, json, sqlite3, uuid, base64, re
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
load_dotenv(ROOT/'.env')
OUT=ROOT/'generated'; OUT.mkdir(exist_ok=True)
DB=ROOT/'omniai.db'
MAX_MB=int(os.getenv('MAX_UPLOAD_MB','15'))

def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    return c

@asynccontextmanager
async def lifespan(app):
    with db() as c:
        c.execute('CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, session TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created TEXT NOT NULL)')
    yield

app=FastAPI(title='OmniAI', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')
app.mount('/generated', StaticFiles(directory=OUT), name='generated')

@app.get('/')
def home(): return FileResponse(ROOT/'static'/'index.html')

@app.get('/api/status')
def status():
    return {'chat':bool(os.getenv('LLM_API_KEY')),'image':bool(os.getenv('IMAGE_API_KEY') or os.getenv('LLM_API_KEY')),'video':bool(os.getenv('REPLICATE_API_TOKEN') and os.getenv('REPLICATE_VIDEO_MODEL')),'local_image_edit':True,'pdf':True}

async def read_upload(f:UploadFile, allowed:set):
    ext=Path(f.filename or '').suffix.lower()
    if ext not in allowed: raise HTTPException(400,'Unsupported file type')
    b=await f.read(MAX_MB*1024*1024+1)
    if len(b)>MAX_MB*1024*1024: raise HTTPException(413,'File too large')
    return b,ext

def save_msg(session,role,content):
    with db() as c:
        c.execute('INSERT INTO messages VALUES (?,?,?,?,?)',(str(uuid.uuid4()),session,role,content,datetime.now(timezone.utc).isoformat()))

def intent(text):
    s=text.lower()
    if re.search(r'\b(image|picture|photo|draw|illustration|poster|tasveer)\b',s) and re.search(r'\b(create|generate|make|banao|bana|draw)\b',s):return 'image'
    if re.search(r'\b(video|clip|animation)\b',s) and re.search(r'\b(create|generate|make|banao|bana)\b',s):return 'video'
    if re.search(r'\b(solve|equation|homework|doubt|calculate|explain)\b',s):return 'doubt'
    return 'chat'


async def llm(messages):
    key = os.getenv('LLM_API_KEY')
    if not key:
        raise HTTPException(
            503, 'Chat model not configured. Set LLM_API_KEY.'
        )

    base = os.getenv(
        'LLM_BASE_URL',
        'https://api.openai.com/v1'
    ).rstrip('/')
    url = base + '/chat/completions'
    model = os.getenv('LLM_MODEL', 'gpt-4o-mini')

    async with httpx.AsyncClient(timeout=90) as client:
        try:
            r = await client.post(
                url,
                headers={'Authorization': f'Bearer {key}'},
                json={
                    'model': model,
                    'messages': messages,
                    'temperature': 0.6
                }
            )

            if r.is_error:
                # Log provider status and a short error body.
                print(
                    f'LLM provider returned HTTP {r.status_code}: '
                    f'{r.text[:1000]}',
                    flush=True
                )
                raise HTTPException(
                    502,
                    f'Chat provider returned HTTP {r.status_code}. '
                    'Check Render logs for details.'
                )

            data = r.json()
            return data['choices'][0]['message']['content']

        except HTTPException:
            raise
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            print(
                f'LLM request failed: {type(e).__name__}: {str(e)[:500]}',
                flush=True
            )
            raise HTTPException(
                502,
                'Chat provider request failed. Check Render logs.'
            )
@app.post('/api/chat')
async def chat(session:str=Form('default'),prompt:str=Form(...),file:UploadFile|None=File(None)):
    if not prompt.strip(): raise HTTPException(400,'Message required')
    if len(prompt)>12000: raise HTTPException(400,'Message too long')
    system='You are OmniAI, a helpful multilingual AI assistant. Explain academic doubts step by step. Be accurate, admit uncertainty, and do not claim to have generated files when you have not.'
    content=[]; attachment=''
    if file:
        data,ext=await read_upload(file,{'.pdf','.png','.jpg','.jpeg','.webp','.txt'})
        if ext=='.pdf':
            try:
                reader=PdfReader(io.BytesIO(data)); attachment='\n\nDocument text:\n'+'\n'.join((p.extract_text() or '') for p in reader.pages[:15])[:22000]
            except Exception: raise HTTPException(400,'Unable to read PDF')
        elif ext=='.txt':attachment='\n\nUploaded text:\n'+data.decode('utf-8',errors='replace')[:22000]
        else:
            try:
                im=Image.open(io.BytesIO(data)).convert('RGB'); im.thumbnail((1200,1200)); buf=io.BytesIO(); im.save(buf,format='JPEG',quality=80)
            except Exception: raise HTTPException(400,'Invalid image')
            content=[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()}}]
    with db() as c:
        rows=c.execute('SELECT role,content FROM messages WHERE session=? ORDER BY created DESC LIMIT 12',(session[:100],)).fetchall()
    messages=[{'role':'system','content':system}]+[{'role':r['role'],'content':r['content']} for r in reversed(rows)]
    messages.append({'role':'user','content':content if content else prompt+attachment})
    answer=await llm(messages)
    save_msg(session[:100],'user',prompt+(f' [attached {file.filename}]' if file else ''))
    save_msg(session[:100],'assistant',answer)
    return {'answer':answer,'intent':intent(prompt)}

@app.get('/api/history/{session}')
def history(session:str):
    with db() as c:rows=c.execute('SELECT role,content,created FROM messages WHERE session=? ORDER BY created LIMIT 100',(session[:100],)).fetchall()
    return {'messages':[dict(r) for r in rows]}
 
@app.delete('/api/history/{session}')
def delete_history(session: str):
    session = session[:100]
    with db() as c:
        c.execute('DELETE FROM messages WHERE session = ?', (session,))
    return {'ok': True, 'deleted_session': session}
    





@app.post('/api/image/generate')
async def image_generate(
    prompt: str = Form(...),
    style: str = Form('realistic'),
    size: str = Form('1024x1024')
):
    key = os.getenv('IMAGE_API_KEY') or os.getenv('LLM_API_KEY')

    if not key:
        raise HTTPException(
            503,
            'Set IMAGE_API_KEY in Render Environment'
        )

    model = os.getenv(
        'IMAGE_MODEL', 'gemini-2.5-flash-image'
    )

    styles = {
        'realistic': 'photorealistic, natural lighting',
        'anime': 'anime illustration',
        'digital art': 'detailed digital artwork',
        '3D render': 'high-quality 3D render',
        'watercolor': 'watercolor painting'
    }

    ratios = {
        '1024x1024': '1:1',
        '1536x1024': '3:2',
        '1024x1536': '2:3'
    }

    if size not in ratios:
        raise HTTPException(400, 'Invalid image size')

    final_prompt = (
        prompt[:2800] + '\nStyle: ' +
        styles.get(style, styles['realistic'])
    )

    url = (
        'https://generativelanguage.googleapis.com'
        '/v1beta/models/' + model + ':generateContent'
    )

    payload = {
        'contents': [{
            'parts': [{'text': final_prompt}]
        }],
        'generationConfig': {
            'responseModalities': ['TEXT', 'IMAGE'],
            'imageConfig': {
                'aspectRatio': ratios[size]
            }
        }
    }

    async with httpx.AsyncClient(timeout=180) as client:
        try:
            response = await client.post(
                url,
                headers={
                    'x-goog-api-key': key,
                    'Content-Type': 'application/json'
                },
                json=payload
            )

            if response.status_code == 429:
                raise HTTPException(
                    429,
                    'Gemini image quota exceeded. '
                    'Check API usage and rate limits.'
                )

            if response.status_code != 200:
                raise HTTPException(
                    502,
                    'Gemini image API error: ' +
                    response.text[:350]
                )

            data = response.json()
            candidates = data.get('candidates', [])

            if not candidates:
                raise HTTPException(
                    502,
                    'No image generated: ' +
                    str(data.get('promptFeedback', ''))[:200]
                )

            parts = candidates[0].get(
                'content', {}
            ).get('parts', [])

            for part in parts:
                image_data = part.get('inlineData')

                if image_data and image_data.get('data'):
                    raw = base64.b64decode(
                        image_data['data']
                    )

                    with Image.open(io.BytesIO(raw)) as im:
                        output = im.convert('RGB')
                        name = str(uuid.uuid4()) + '.png'
                        output.save(OUT / name)

                    return {
                        'url': '/generated/' + name,
                        'style': style,
                        'size': size
                    }

            raise HTTPException(
                502,
                'Gemini returned no image. Check model access.'
            )

        except httpx.HTTPError as e:
            raise HTTPException(
                502,
                'Gemini connection error: ' + str(e)[:200]
            )
        except (ValueError, OSError, KeyError) as e:
            raise HTTPException(
                502,
                'Image processing error: ' + str(e)[:200]
            )

@app.post('/api/image/edit')
async def image_edit(file:UploadFile=File(...),operation:str=Form('enhance')):
    raw,_=await read_upload(file,{'.png','.jpg','.jpeg','.webp'})
    try:im=Image.open(io.BytesIO(raw)).convert('RGB'); im.thumbnail((2048,2048))
    except Exception:raise HTTPException(400,'Invalid image')
    if operation=='enhance': im=ImageEnhance.Contrast(ImageEnhance.Color(ImageEnhance.Sharpness(im).enhance(1.5)).enhance(1.1)).enhance(1.1)
    elif operation=='grayscale':im=ImageOps.grayscale(im).convert('RGB')
    elif operation=='blur':im=im.filter(ImageFilter.GaussianBlur(4))
    elif operation=='rotate':im=im.rotate(90,expand=True)
    elif operation=='mirror':im=ImageOps.mirror(im)
    else:raise HTTPException(400,'Invalid operation')
    name=str(uuid.uuid4())+'.png'; im.save(OUT/name)
    return {'url':'/generated/'+name,'note':'Local image transformation. Generative editing requires a separate compatible model.'}

@app.post('/api/video/generate')
async def video_generate(prompt:str=Form(...)):
    token=os.getenv('REPLICATE_API_TOKEN'); model=os.getenv('REPLICATE_VIDEO_MODEL')
    if not token or not model:raise HTTPException(503,'Video API not configured. Set REPLICATE_API_TOKEN and REPLICATE_VIDEO_MODEL.')
    # Replicate model versions and inputs vary. REPLICATE_VIDEO_MODEL must be owner/model.
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r=await client.post(f'https://api.replicate.com/v1/models/{model}/predictions',headers={'Authorization':f'Bearer {token}','Content-Type':'application/json','Prefer':'wait=5'},json={'input':{'prompt':prompt[:2000]}})
            r.raise_for_status(); data=r.json()
            return {'id':data.get('id'),'status':data.get('status'),'output':data.get('output'),'note':'Video generation is asynchronous. Check prediction status.'}
        except httpx.HTTPError as e:raise HTTPException(502,f'Video provider error: {str(e)[:250]}')

@app.get('/api/video/status/{prediction_id}')
async def video_status(prediction_id:str):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{5,100}',prediction_id):raise HTTPException(400,'Invalid prediction ID')
    token=os.getenv('REPLICATE_API_TOKEN')
    if not token:raise HTTPException(503,'Video API not configured')
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r=await client.get(f'https://api.replicate.com/v1/predictions/{prediction_id}',headers={'Authorization':f'Bearer {token}'})
            r.raise_for_status(); data=r.json()
            return {'status':data.get('status'),'output':data.get('output'),'error':data.get('error')}
        except httpx.HTTPError as e:raise HTTPException(502,str(e)[:250])
