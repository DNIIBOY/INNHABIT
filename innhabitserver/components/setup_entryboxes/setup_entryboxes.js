document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("drawingCanvas");
    if (!canvas) return; // Ensures the script runs only when the canvas exists

    const img = document.getElementById("backgroundImage");

    let boxes = {
        entry_box: { "entry_top_left_x": 0, "entry_top_left_y": 0, "entry_bottom_right_x": 0, "entry_bottom_right_y": 0 },
        exit_box: { "exit_top_left_x": 0, "exit_top_left_y": 0, "exit_bottom_right_x": 0, "exit_bottom_right_y": 0 }
    };

    let dragBoxBeginCoord = {x_top: 0, y_top: 0, x_btm: 0, y_btm: 0}
    let dragMouseBeginCoord = [0,0]

    let selectedCorner = null;
    let closestCorner = null;
    let boxBeingDragged = null;

    const minBoxSize = 50

    const cornerSelectDistance = 25;

    const cornerHalfBoxSize = 5;
    const cornerHalfBoxSizeSelected = 8;

    const originalWidth = img.naturalWidth;
    const originalHeight = img.naturalHeight;

    const ctx = canvas.getContext("2d");

    function resizeCanvas() {
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        drawRectangles();
    }

    window.addEventListener("load", resizeCanvas);
    window.addEventListener("resize", resizeCanvas);

    function remapWidth(value) {
        return canvas.width * value / originalWidth;
    }

    function remapHeight(value) {
        return canvas.height * value / originalHeight;
    }

    function remapWidthToOriginal(value) {
        return originalWidth * value / canvas.width;
    }

    function remapHeightToOriginal(value) {
        return originalHeight * value / canvas.height;
    }


    function drawRectangles() {
        const entry_box_color = "rgba(0.0,200,0.5)";
        const exit_box_color = "rgba(200,0,0,0.5)";

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        Object.keys(boxes).forEach((boxKey) => {
            const values = boxes[boxKey];
            const { x_top, y_top, x_btm, y_btm } = values;

            // Calculate rectangle size
            const size_x = x_btm - x_top;
            const size_y = y_btm - y_top;

            // Draw the rectangle
            ctx.fillRect(remapWidth(x_top), remapHeight(y_top), remapWidth(size_x), remapHeight(size_y));


            ctx.fillStyle = "rgba(190,190,190)";
            ctx.lineWidth = 1; // Thickness of the rim
            ctx.strokeStyle = "rgba(60,60,60)"; // Rim color

            ctx.fillRect(
                remapWidth(x_top) - cornerHalfBoxSize, 
                remapHeight(y_top) - cornerHalfBoxSize, 
                cornerHalfBoxSize * 2,
                cornerHalfBoxSize * 2
            )

            ctx.strokeRect(
                remapWidth(x_top) - cornerHalfBoxSize, 
                remapHeight(y_top) - cornerHalfBoxSize, 
                cornerHalfBoxSize * 2,
                cornerHalfBoxSize * 2
            )

            ctx.fillRect(
                remapWidth(x_top) - cornerHalfBoxSize, 
                remapHeight(y_btm) - cornerHalfBoxSize, 
                cornerHalfBoxSize * 2,
                cornerHalfBoxSize * 2
            )

            ctx.strokeRect(
                remapWidth(x_top) - cornerHalfBoxSize, 
                remapHeight(y_btm) - cornerHalfBoxSize, 
                cornerHalfBoxSize * 2,
                cornerHalfBoxSize * 2
            )

            ctx.fillRect(
                remapWidth(x_btm) - cornerHalfBoxSize, 
                remapHeight(y_top) - cornerHalfBoxSize, 
                cornerHalfBoxSize * 2,
                cornerHalfBoxSize * 2
            )

            ctx.strokeRect(
                remapWidth(x_btm) - cornerHalfBoxSize, 
                remapHeight(y_top) - cornerHalfBoxSize, 
                cornerHalfBoxSize * 2,
                cornerHalfBoxSize * 2
            )

            ctx.fillRect(
                remapWidth(x_btm) - cornerHalfBoxSize, 
                remapHeight(y_btm) - cornerHalfBoxSize, 
                cornerHalfBoxSize * 2,
                cornerHalfBoxSize * 2
            )

            ctx.strokeRect(
                remapWidth(x_btm) - cornerHalfBoxSize, 
                remapHeight(y_btm) - cornerHalfBoxSize, 
                cornerHalfBoxSize * 2,
                cornerHalfBoxSize * 2
            )


            ctx.fillStyle = "rgba(190,190,190,1)";
            if (closestCorner != null) {
                switch (closestCorner) {
                    case "top-left":
                        ctx.fillRect(
                            remapWidth(x_top) - cornerHalfBoxSizeSelected, 
                            remapHeight(y_top) - cornerHalfBoxSizeSelected, 
                            cornerHalfBoxSizeSelected * 2,
                            cornerHalfBoxSizeSelected * 2
                        )

                        ctx.strokeRect(
                            remapWidth(x_top) - cornerHalfBoxSizeSelected, 
                            remapHeight(y_top) - cornerHalfBoxSizeSelected, 
                            cornerHalfBoxSizeSelected * 2,
                            cornerHalfBoxSizeSelected * 2
                        )
                        break;
                    case "top-right":
                        ctx.fillRect(
                            remapWidth(x_btm) - cornerHalfBoxSizeSelected, 
                            remapHeight(y_top) - cornerHalfBoxSizeSelected, 
                            cornerHalfBoxSizeSelected * 2,
                            cornerHalfBoxSizeSelected * 2
                        )

                        ctx.strokeRect(
                            remapWidth(x_btm) - cornerHalfBoxSizeSelected, 
                            remapHeight(y_top) - cornerHalfBoxSizeSelected, 
                            cornerHalfBoxSizeSelected * 2,
                            cornerHalfBoxSizeSelected * 2
                        )
                        break;
                    case "bottom-left":
                        ctx.fillRect(
                            remapWidth(x_top) - cornerHalfBoxSizeSelected, 
                            remapHeight(y_btm) - cornerHalfBoxSizeSelected, 
                            cornerHalfBoxSizeSelected * 2,
                            cornerHalfBoxSizeSelected * 2
                        )

                        ctx.strokeRect(
                            remapWidth(x_top) - cornerHalfBoxSizeSelected, 
                            remapHeight(y_btm) - cornerHalfBoxSizeSelected, 
                            cornerHalfBoxSizeSelected * 2,
                            cornerHalfBoxSizeSelected * 2
                        )
                        break;
                    case "bottom-right":
                        ctx.fillRect(
                            remapWidth(x_btm) - cornerHalfBoxSizeSelected, 
                            remapHeight(y_btm) - cornerHalfBoxSizeSelected, 
                            cornerHalfBoxSizeSelected * 2,
                            cornerHalfBoxSizeSelected * 2
                        )
                        ctx.strokeRect(
                            remapWidth(x_btm) - cornerHalfBoxSizeSelected, 
                            remapHeight(y_btm) - cornerHalfBoxSizeSelected, 
                            cornerHalfBoxSizeSelected * 2,
                            cornerHalfBoxSizeSelected * 2
                        )
                        break;
                }
                ctx.lineWidth = 1; // Thickness of the rim
                ctx.strokeStyle = "black"; // Rim color
                ctx.stroke(); // Draw the rim
            }
        });
    }


    function updateFormFields() {
        Object.keys(boxes).forEach((boxKey) => {
            Object.keys(boxes[boxKey]).forEach((key) => {
                document.getElementById(key).value = parseInt(boxes[boxKey][key]);
            });
        });
    }

    function calculateDistance(x1, y1, x2, y2) {
        return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    }

    function EndBoxManipulation() {
        isDraggingBox = null;
        selectedCorner = null;

        // Snaps to edge
        let snapDistance = 20;

        Object.keys(boxes).forEach((boxKey) => {
            const values = boxes[boxKey];
            const { x_top, y_top, x_btm, y_btm } = values;
        
            // Snap to edge on corresponding side
            if (x_top < snapDistance) {
                box.x_top = 0;
            }
            if (y_top < snapDistance) {
                box.y_top = 0;
            }
            if (x_btm > originalWidth - snapDistance) {
                box.x_btm = originalWidth;
            }
            if (y_btm > originalHeight - snapDistance) {
                box.y_btm = originalHeight;
            }

            // Stop box from squishing to the sides
            if (x_top > originalWidth - minBoxSize) {
                box.x_top = originalWidth - minBoxSize;
            }
            if (y_top > originalHeight - minBoxSize) {
                box.y_top = originalHeight - minBoxSize;
            }
            if (x_btm < minBoxSize) {
                box.x_btm = minBoxSize;
            }
            if (y_btm < minBoxSize) {
                box.y_btm = minBoxSize;
            }
        });
    }

    canvas.addEventListener("pointermove", function(event) {
        const rect = canvas.getBoundingClientRect();
        const x = remapWidthToOriginal(event.clientX - rect.left);
        const y = remapHeightToOriginal(event.clientY - rect.top);

        Object.keys(boxes).forEach((boxKey) => {
            const values = boxes[boxKey];
            const { x_top, y_top, x_btm, y_btm } = values;

            if (selectedCorner == null) {
                const distances = {
                    "top-left": calculateDistance(x, y, x_top, y_top),
                    "top-right": calculateDistance(x, y, x_btm, y_top),
                    "bottom-left": calculateDistance(x, y, x_top, y_btm),
                    "bottom-right": calculateDistance(x, y, x_btm, y_btm)
                };

                closestCorner = Object.keys(distances).reduce((minCorner, currentCorner) => {
                    return distances[currentCorner] < distances[minCorner] ? currentCorner : minCorner;
                });
                

                if (distances[closestCorner] > cornerSelectDistance) {
                    closestCorner = null;
                }
            }
            
            switch (selectedCorner) {
                case "top-left":
                    if (x > box.x_btm - minBoxSize) {
                        box.x_top = Math.max(box.x_btm-minBoxSize,0);
                    } else {
                        box.x_top = x;
                    }
                    if (y > box.y_btm - minBoxSize) {
                        box.y_top = Math.max(y_btm-minBoxSize,0);
                    } else {
                        box.y_top = y;
                    }
                    break;
                case "top-right":
                    if (x < box.x_top + minBoxSize) {
                        box.x_btm = Math.min(x_top+minBoxSize,originalWidth);
                    } else {
                        box.x_btm = x;
                    }
                    if (y > box.y_btm - minBoxSize) {
                        box.y_top = Math.max(y_btm-minBoxSize,0);
                    } else {
                        box.y_top = y;
                    }
                    break;
                case "bottom-left":
                    if (x > box.x_btm - minBoxSize) {
                        box.x_top = Math.max(x_btm-minBoxSize,0);
                    } else {
                        box.x_top = x;
                    }
                    if (y < box.y_top + minBoxSize) {
                        box.y_btm = Math.min(y_top+minBoxSize,originalHeight);
                    } else {
                        box.y_btm = y;
                    }
                    break;
                case "bottom-right":
                    if (x < box.x_top + minBoxSize) {
                        box.x_btm = Math.min(x_top+minBoxSize,originalWidth);
                    } else {
                        box.x_btm = x;
                    }
                    if (y < box.y_top + minBoxSize) {
                        box.y_btm = Math.min(y_top+minBoxSize,originalHeight);
                    } else {
                        box.y_btm = y;
                    }
                    break;
            }

            if (isDraggingBox == true) {
                box.x_top = dragBoxBeginCoord.x_top + (x - dragMouseBeginCoord[0]);
                box.x_btm = dragBoxBeginCoord.x_btm + (x - dragMouseBeginCoord[0]);
                box.y_top = dragBoxBeginCoord.y_top + (y - dragMouseBeginCoord[1]);
                box.y_btm = dragBoxBeginCoord.y_btm + (y - dragMouseBeginCoord[1]);
            }
        });
        updateFormFields()
        drawRectangles();
    });

    canvas.addEventListener("pointerdown", function(event) {
        selectedCorner = closestCorner;
        boxDrag = null;

        // box drag
        if (selectedCorner == null) {
            const rect = canvas.getBoundingClientRect();
            const x = remapWidthToOriginal(event.clientX - rect.left);
            const y = remapHeightToOriginal(event.clientY - rect.top);

            Object.keys(boxes).forEach((boxKey) => {
                const values = boxes[boxKey];
                const { x_top, y_top, x_btm, y_btm } = values;

                if(x_btm > x && x > x_top && y_btm > y && y > y_top){
                    boxBeingDragged = boxKey;
                    dragBoxBeginCoord = {x_top: x_top, y_top: y_top, x_btm: x_btm, y_btm: y_btm};
                    dragMouseBeginCoord = [x,y]
                }
            });
        }
    });

    canvas.addEventListener("pointerup", function(event) {
        EndBoxManipulation();

        updateFormFields();
        drawRectangles();
    });

    canvas.addEventListener("pointerleave", function(event) {
        EndBoxManipulation();

        updateFormFields();
        drawRectangles();
    });

    updateFormFields();
    drawRectangles();
});
