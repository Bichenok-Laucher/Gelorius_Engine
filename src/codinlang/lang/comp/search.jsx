function search(e) {
    let searched = document.getElementById("search").value.trim();
  if (searched !== "") {
    let text = document.getElementById("text").innerHTML;
    let re = new RegExp(searched,"g"); // search for all instances
        let newText = text.replace(re, `<mark>${searched}</mark>`);
        document.getElementById("text").innerHTML = newText;
  }
}