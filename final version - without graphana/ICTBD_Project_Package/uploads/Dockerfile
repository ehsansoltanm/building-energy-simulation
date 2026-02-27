FROM jupyter/scipy-notebook

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /home/jovyan/work

CMD ["start-notebook.sh"]
