function highlight_words(word) {
    const page = document.body.innerHTML;
    document.body.innerHTML = page.replace(new RegExp(word, "gi"), (match) => `<mark>${match}</mark>`);
}