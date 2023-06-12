function stylizeHighlightedString() {

    var text = window.getSelection();

    // For diagnostics
    var start = text.anchorOffset;
    var end = text.focusOffset - text.anchorOffset;

    range = window.getSelection().getRangeAt(0);

    var selectionContents = range.extractContents();
    var span = document.createElement("span");

    span.appendChild(selectionContents);

    span.style.backgroundColor = "yellow";
    span.style.color = "black";

    range.insertNode(span);
}