
.PHONY: force-build

all: Thesis.pdf propositions.pdf Thesis_print.pdf

force-build:


#figs: figs/development_model_without_papers.pdf figs/development_model_just_papers.pdf

figs: 

# figs/development_model_without_papers.pdf:
# 	rsvg-convert -f pdf -o  figs/development_model_without_papers.pdf figs/development_model_without_papers.svg

# figs/development_model_just_papers.pdf:
# 	rsvg-convert -f pdf -o  figs/development_model_just_papers.pdf figs/development_model_just_papers.svg

Thesis.pdf: figs force-build
	latexmk -xelatex Thesis
	makeglossaries Thesis

propositions.pdf: force-build
	latexmk -xelatex propositions

Thesis_print.pdf: Thesis.pdf
	gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dEmbedAllFonts=true -sOutputFile=dissertation_print.pdf -f dissertation.pdf

clean:
	rubber --clean Thesis.tex
	rm -f Thesis.pdf
