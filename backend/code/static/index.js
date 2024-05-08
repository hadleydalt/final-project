let current = "link1"
const link_ids = ["link1", "link2", "link3"]

function fetch_test() {
    fetch('/fetchtest').then(
        response => response.json()
    ).then(function(data) {
        document.getElementById("to_change").innerHTML = data['hello']
    })
}

function set_current(id) {
    current = id
    match_current()
}

function match_current() {
    link_ids.forEach((id) => {
        if (id == current) {
            document.getElementById(id).style.color = "black"
            document.getElementById(id).style.background = "white"
        } else {
            document.getElementById(id).style.color = "white"
            document.getElementById(id).style.background = "transparent"
        }
    })
}

function setup_listeners() {
    link_ids.forEach((id) => {
        document.getElementById(id).addEventListener("click", ()=>set_current(id))
    })
}

setup_listeners()
match_current()