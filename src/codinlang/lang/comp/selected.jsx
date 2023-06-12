function styleSelected() {
    bg = document.createElement("span");
    bg.style.backgroundColor = "yellow";
    window.getSelection().getRangeAt(0).surroundContents(bg);
  }