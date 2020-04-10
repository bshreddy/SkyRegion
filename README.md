# Sky Region AI Inference Server on Flask

  * Server IP: `<your-computer-ip>:8001`

  * ## Installation
    
    * Install required packages <br />
      `pip3 install -r requirements.txt`

    * Download models <br />
      `python3 download_models.py`

    * Run Server <br />
      `python3 app.py`

  * ## Installation (Docker) (Recommended)

    * To Build: <br />
      `docker build -t sky-region -f docker/Dockerfile .`
    
    * To Run:<br />
      `docker run -d --rm -p 8001:8000 --name sky-region sky-region`