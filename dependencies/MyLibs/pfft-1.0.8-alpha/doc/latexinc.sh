# Copyright (C) 2011 The ScaFaCoS project
#  
# This file is part of ScaFaCoS.
#  
# ScaFaCoS is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#  
#  ScaFaCoS is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser Public License for more details.
#  
#  You should have received a copy of the GNU Lesser Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
#

# This file is included from latex-manual.sh and sets the default
# paths for latexing the documentation.
if test -z "$TEXINPUTS"; then
    TEXINPUTS=.:/lhome/ific/m/marlu/cosmolattice/dependencies/MyLibs/pfft-1.0.8-alpha/doc:
else
    TEXINPUTS=.:/lhome/ific/m/marlu/cosmolattice/dependencies/MyLibs/pfft-1.0.8-alpha/doc:$TEXINPUTS:
fi

if test -z "$BIBINPUTS"; then
    BIBINPUTS=.:/lhome/ific/m/marlu/cosmolattice/dependencies/MyLibs/pfft-1.0.8-alpha/doc:
else
    BIBINPUTS=.:/lhome/ific/m/marlu/cosmolattice/dependencies/MyLibs/pfft-1.0.8-alpha/doc:$BIBINPUTS:
fi

export TEXINPUTS BIBINPUTS

PDFLATEX=/bin/pdflatex
BIBTEX=/bin/bibtex
MAKEINDEX=/bin/makeindex
