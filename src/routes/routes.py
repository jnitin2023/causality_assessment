from fastapi import APIRouter

from src.routes import dataprocessing, dataprocessing_camelot, dataprocessing_pypdf, dataprocessing_pymupdf, dataprocessing_nlp, assessment


api_router = APIRouter()

api_router.include_router(dataprocessing.router, prefix="/data-process", tags=["dataprocessing_pdfplumber"])
api_router.include_router(dataprocessing_pypdf.router, prefix="/data-process", tags=["dataprocessing_pypdf"])
# api_router.include_router(dataprocessing_camelot.router, prefix="/data-process", tags=["dataprocessing_camelot"])
api_router.include_router(dataprocessing_pymupdf.router, prefix="/upload", tags=["dataprocessing_pymupdf"])
api_router.include_router(dataprocessing_nlp.router, prefix="/extract_pdf_data", tags=["dataprocessing_nlp"])
api_router.include_router(assessment.router, prefix="/assessment", tags=["assessment"])
