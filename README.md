# React + FastAPI demo

This project demonstrates a real-time interactive coordinate system ($y = kx$) using a FastAPI backend for calculations and a React (Vite + TypeScript) frontend for visualization.

## Project structure
 - simple_function: frontend code
 - backend: FastAPI code

## Prerequisites: 
### 1. Install Node.js (Optional)
If you do not have Node.js installed, download the LTS version from the official site. Node includes npm, which is required to manage frontend dependencies.
 - Download: nodejs.org
 - Verify: Run `node -v` and `npm -v` in your terminal.

### 2. Python environment
#### Step 1. Ensure you have Python 3.9+ installed
 - Verify: `python --version`

#### Step 2. Install dependencies
 Navigate to backend directory
  - Run `pip install -r requirements.txt`

## Get the project running:
Nagivate to the backend directory, run:  

`unicorn main:app`  

Now you can see it in action!

## Making changes to the frontend?


### Development Setup
To see changes in real-time as you code, you will run the frontend and backend servers simultaneously.
#### Step 1. Backend (FastAPI)
##### 1. Navigate to backend directory
##### 2. Start the server with "Hot Reload" enabled:
- `uvicorn main:app --reload`
- The API will be available at: http://localhost:8000
- Swagger Documentation: http://localhost:8000/docs
#### Step 2. Frontend (React + Vite)
##### 1. Navigate to frontend directory
##### 2. Install dependencies:
- `npm install`
##### 3. Start the Vite development server:
- `npm run dev`
- The UI will be available at: http://localhost:5173
- Note: Any changes you make to App.tsx will reflect instantly in the browser.


#### Step 2. Install dependencies
 Navigate to backend directory
  - Run `pip install -r requirements.txt`

### Production Build & Integration
##### 1. Build the frontend:
 - Inside the frontend directory, run:
   - `npm run build`
   - This generates a dist/ folder containing optimized HTML, CSS, and JavaScript.
##### 2. Move the files
- Move or copy the contents of the dist/ folder into your backend/ directory.
##### 3. Connect FastAPI to the Frontend
Ensure your main.py includes the following lines to serve the static files:  

```
from fastapi.staticfiles import StaticFiles

# Mount the 'dist' directory to the root URL
app.mount("/", StaticFiles(directory="dist", html=True), name="static")
```
##### 4. Run the final product
Nagivate to the backend directory, run:  

`unicorn main:app`  

Now you can see the updates in action!

 
