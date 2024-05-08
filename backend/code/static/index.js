function fetch_test() {
    fetch('/fetchtest').then(
        response => response.json()
    ).then(function(data) {
        document.getElementById("to_change").innerHTML = data['hello']
    })
}