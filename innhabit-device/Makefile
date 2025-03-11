CXX := g++
CXXFLAGS := -g -Wall $(shell pkg-config --cflags opencv4) -I/usr/include/opencv4
LDFLAGS := $(shell pkg-config --libs opencv4) -lcurl
TARGET := main
SRC := main.cpp
OBJ := $(SRC:.cpp=.o)

# List of files to download
URLS := https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.weights \
        https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7-tiny.cfg \
        https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
FILES := yolov7-tiny.weights yolov7-tiny.cfg coco.names

all: download compile_commands.json $(TARGET)

download:
	@i=1; for file in $(FILES); do \
		url=$$(echo $(URLS) | cut -d' ' -f$$i); \
		if [ ! -f $$file ]; then \
			echo "Downloading $$file..."; \
			wget -O $$file $$url; \
		else \
			echo "$$file already exists, skipping download."; \
		fi; \
		i=$$((i+1)); \
	done

compile_commands.json:
	@command -v bear >/dev/null 2>&1 && bear -- make $(TARGET) || echo "Bear not installed. Skipping compile_commands.json generation."

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) compile_commands.json
