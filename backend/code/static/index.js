function fetch_test() {
    fetch('/fetchtest').then(
        response => response.json()
    ).then(function(data) {
        console.log(data)
        document.getElementById("to_change").innerHTML = data['hello']
    })
}