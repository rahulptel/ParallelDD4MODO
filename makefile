# --- SYSTEM ---

SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic
BASISDIR   = /opt
PLATFORM   = linux64


# --- DIRECTORIES ---

CCC = g++
NVCC ?= nvcc
USE_CUDA ?= 0

BOOSTDIR   = $(BASISDIR)/boost

GUROBIDIR = /opt/gurobi/$(PLATFORM)
GUROBIINCDIR = $(GUROBIDIR)/include
GUROBILIBDIR = $(GUROBIDIR)/lib


# --- FLAGS ---

CCOPT = -std=c++11 -m64 -fPIC -fno-strict-aliasing -fexceptions -Wno-deprecated-declarations -Wno-ignored-attributes

# --- OPTIMIZATION FLAGS ---

DEBUG_OPT = -DNDEBUG -O3
#DEBUG_OPT = -g3 -O0

#PROF = -pg
PROF =


# Flags
CFLAGS  = -I$(BOOSTDIR)/include -c $(PROF) $(DEBUG_OPT)
LDFLAGS = -lm -pthread

# Base compilation flags
CFLAGS  += $(CCOPT)

# Uncomment if using Gurobi
#CFLAGS += -I$(GUROBIINCDIR)
#LDFLAGS += -L$(GUROBILIBDIR) -lgurobi_c++ -lgurobi75 -lm 


# --- APPLICATION-SPECIFIC OPTIONS ---

# number of objective functions
NUM_OBJS=3
CFLAGS += -DNOBJS=$(NUM_OBJS)

ifeq ($(USE_CUDA),1)
CFLAGS += -DUSE_CUDA
CUDAFLAGS = -std=c++11 -O3 -DUSE_CUDA -DNOBJS=$(NUM_OBJS) -I$(BOOSTDIR)/include
LDFLAGS += -lcudart
endif


# ---- COMPILE  ----
SRC_DIR   := src
OBJ_DIR   := obj

SRC_DIRS  := $(shell find $(SRC_DIR) -type d)
OBJ_DIRS  := $(addprefix $(OBJ_DIR)/,$(SRC_DIRS))

SOURCES_CPP := $(shell find $(SRC_DIR) -name '*.cpp')
OBJ_FILES   := $(addprefix $(OBJ_DIR)/, $(SOURCES_CPP:.cpp=.o))

ifeq ($(USE_CUDA),1)
CUDA_SOURCES  := $(shell find $(SRC_DIR) -name '*.cu')
CUDA_OBJ_FILES := $(addprefix $(OBJ_DIR)/, $(CUDA_SOURCES:.cu=.o))
OBJ_FILES     += $(CUDA_OBJ_FILES)
vpath %.cu $(SRC_DIRS)
endif

vpath %.cpp $(SRC_DIRS)

# ---- TARGETS ----

EXECUTABLE=multiobj

all: $(EXECUTABLE)

$(EXECUTABLE): makedir $(OBJ_FILES) 
	$(CCC) $(OBJ_FILES) $(LDFLAGS) $(PROF) -o $@

$(OBJ_DIR)/%.o: %.cpp
	$(CCC) $(CFLAGS) $< -o $@

$(OBJ_DIR)/%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

makedir: $(OBJ_DIRS)

$(OBJ_DIRS):
	@mkdir -p $@

clean:
	@rm -rf obj 
	@rm -rf $(EXECUTABLE)
