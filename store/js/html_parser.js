function childLength(path) {
  let ele = document.body;
  for(let i = 0; i < path.length; i++) {
    let childIndex = path[i];
    ele = ele.children[childIndex];
  }
  return ele.children.length;
}

function getStyle(runId, path) {
  let ele = document.body;
  for(let i = 0; i < path.length; i++) {
      let childIndex = path[i];
      ele = ele.children[childIndex];
  }
  window.elementId += 1;
  const className = runId + "-" + window.elementId;
  ele.classList.add(className);
  const styles = window.getComputedStyle(ele);
  const eleStyles = {};
  for(let i = 0; i < styles.length; i++) {
      eleStyles[styles[i]] = styles.getPropertyValue(styles[i]);
  }
  return {id: window.elementId, style: eleStyles, tagName: ele.tagName, bounds: ele.getBoundingClientRect()};
}

function buildTree(runId, path = [], data = {}) {
  const styleData = getStyle(runId, path);
  data['tag'] = styleData['tagName'];
  const filteredStyle = {};    
  Object.entries(styleData['style']).forEach(([key, value]) => {
    if(cssProps.indexOf(key) !== -1) {
      filteredStyle[key] = value
    }
  });
  data['styles'] = filteredStyle;
  data['bound'] = styleData['bounds'];
  data['children'] = [];
  data['id'] = styleData['id']
  const childrenCount = childLength(path);
  for(let i = 0; i < childrenCount; i ++) {
    const childStyle = buildTree(runId, path.concat(i));
    data['children'].push(childStyle);
  }
  return data;
}

window.elementId = 0
window.getStyle = getStyle;
window.childLength = childLength;
window.buildTree = buildTree;