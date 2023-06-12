import React from 'react';
interface HighlightProps {
  text: string;
  searchTerm: string;
  highlightStyle?: React.CSSProperties;
}
const defaultHighlightStyle: React.CSSProperties = {
  backgroundColor: 'yellow',
};
const Highlight: React.FC<HighlightProps> = ({
  text,
  searchTerm,
  highlightStyle = defaultHighlightStyle,
}) => {
  if (!searchTerm) {
    return <span>{text}</span>;
  }
  const regex = new RegExp(`(${searchTerm})`, 'gi');
  const parts = text.split(regex);
  return (
    <span>
      {parts.map((part, index) =>
        part.toLowerCase() === searchTerm.toLowerCase() ? (
          <span key={index} style={highlightStyle}>
            {part}
          </span>
        ) : (
          <React.Fragment key={index}>{part}</React.Fragment>
        ),
      )}
    </span>
  );
};

export default Highlight;