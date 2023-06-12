function highlightSelection() {
    var userSelection = window.getSelection();
    for(var i = 0; i < userSelection.rangeCount; i++) {
        highlightRange(userSelection.getRangeAt(i));
    }
    
}
    
    function highlightRange(range) {
        var newNode = document.createElement("span");
        newNode.setAttribute(
           "style",
           "background-color: yellow; display: inline;"
        );
        range.surroundContents(newNode);
    }