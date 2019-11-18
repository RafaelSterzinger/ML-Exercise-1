from pdflatex import PDFLaTeX
import cpu_performance
import student_performance

pdfl = PDFLaTeX.from_texfile("report.tex")
pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)
