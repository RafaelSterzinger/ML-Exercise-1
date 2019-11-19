from pdflatex import PDFLaTeX
import cpu_performance
import student_performance
import online_news_popularity
import bike_sharing

pdfl = PDFLaTeX.from_texfile("report.tex")
pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)
