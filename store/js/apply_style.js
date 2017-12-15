function applyStyles(style_text) {
    const head = document.head;
    const style = document.createElement('style');
    style.type = 'text/css';
    style.appendChild(document.createTextNode(style_text));
    head.appendChild(style);
}

window.applyStyles = applyStyles;