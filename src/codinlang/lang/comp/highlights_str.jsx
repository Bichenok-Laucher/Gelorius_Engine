highlightStr(value, regex, document.getELementById('searchable-div');

const highlightStr = (value, regex, node) => {
        node.childNodes.forEach((childNode) => {
            if (childNode.nodeValue && regex.test(childNode.nodeValue)) {
                const highLightWrapper = document.createElement('span');
                highLightWrapper.classList.add("highlight-wrapper-class");

                childNode.replaceWith(highLightedWrapper);

                highLightWrapper.innerHTML = childNode.nodeValue.replace(regex, `<span class="highlight">${value}</span>`);
            }
            this.highlightStr(value, regex, childNode);
        });
    }