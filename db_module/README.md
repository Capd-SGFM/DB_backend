### connect_sqlalchemy_engine.py ###
PostgreSQL DB와 연결할 수 있는 Engine 객체를 만듦

# def get_sync_engine():
SQLAlchemy의 동기 방식 엔진 생성
Pandas, Streamlit, Jupyter Notebook 같은 분석/대시보드 환경에서 적합

# def get_async_engine():
SQLAlchemy의 비동기 엔진 생성
FastAPI, aiohttp 같은 비동기 웹 프레임워크에서 동시성 처리할 때 적합







# https://4ourfuture.tistory.com/94
# https://sosodev.tistory.com/entry/Python-SQLAlchemy-PostgreSQL-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B2%A0%EC%9D%B4%EC%8A%A4-%EC%97%B0%EB%8F%99%ED%95%98%EA%B8%B0
# https://tofof.tistory.com/108
# https://velog.io/@newnew_daddy/PYTHON02