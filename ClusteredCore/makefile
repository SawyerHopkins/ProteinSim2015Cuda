CC := g++ # This is the main compiler
ARC := ar

SRCDIR := src
BUILDDIR := build
TARGET := bin/libcore.a
 
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g -O3 -fopenmp -std=c++11 -Wall
LIB := -fopenmp
INC := -I include

$(TARGET): $(OBJECTS)
	@echo "Creating static library..."
	@echo " $(ARC) -rcs $(TARGET) $^"; $(ARC) -rcs $(TARGET) $^ 
#@echo " Linking..."
#@echo " $(CC) $^ -o $(TARGET) $(LIB)"; $(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

.PHONY: clean