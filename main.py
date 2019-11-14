from pdflatex import PDFLaTeX
import bike_sharing
import cpu_performance

pdfl = PDFLaTeX.from_texfile("report.tex")
pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)
