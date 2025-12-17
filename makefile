CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Path to LibTorch
LIBTORCH_PATH = /opt/libtorch

INCLUDE = -I$(LIBTORCH_PATH)/include \
          -I$(LIBTORCH_PATH)/include/torch/csrc/api/include

LIBS = -L$(LIBTORCH_PATH)/lib \
       -ltorch -lc10 -lc10_cuda

LDFLAGS = -pthread -Wl,-rpath,$(LIBTORCH_PATH)/lib -ldl

SRC = main.cpp
TARGET = my_program

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDE) $(LIBS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

