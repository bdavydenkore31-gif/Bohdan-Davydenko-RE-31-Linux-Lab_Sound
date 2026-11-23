"""Запуск вебсервісу лабораторної."""
# Цей файл залишається точкою запуску для Python та Uvicorn.
from web.api import app, create_app

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8000)
