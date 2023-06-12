resetHighlight(document.getELementById('searchable-div');

const resetHighlight = (node) => {
        node.childNodes.forEach((childNode) => {
            if ((childNode as HTMLElement).classList && (childNode as HTMLElement).classList.contains('highlight-wrapper-class') && childNode.textContent) {
                childNode.replaceWith(childNode.textContent);
            }
            this.resetHighlight(childNode);
        });
    }