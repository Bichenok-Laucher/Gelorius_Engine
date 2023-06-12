const ExampleComponent = () => {
    const someContent = 'the quick brown fox jumped over the lazy dog';
    const textToHighlight = 'fox';
    return (
        <Highlight text={someContent} searchTerm={textToHighlight} />
    );
}