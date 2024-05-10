function fetch_test() {
    fetch('/fetchtest').then(
        response => response.json()
    ).then(function(data) {
        document.getElementById("to_change").innerHTML = data['hello']
    })
}

document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("upload-form");
    const fileInput = form.querySelector("input[type=file]");

    fileInput.addEventListener("change", function() {
        if (fileInput.files.length > 0) {
            const submitButton = form.querySelector("input[type=submit]");
            submitButton.click();
        }
    });
});