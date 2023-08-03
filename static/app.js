document.getElementById('startButton').addEventListener('click', function() {
// Request the server to start the face recognition system turn on
                                                        fetch('/start_recognition', {method: 'POST'})
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      // the server successfully started the face id process .
    // starting video feed 
    document.getElementById('videoElement').play();

    // starting fetching of face every 5 seconds interval
    setInterval(fetchFaces, 5000);
    } else {
    // handling errors here
    console.error('Failed to start the FAcE recognition:', data.error);
    }
  });
});

function fetchFaces() {
  fetch('/faces')
  .then(response => response.json())
  .then(data => {
    var faceList = document.getElementById('faceList');
    faceList.innerHTML = '';
    data['faces'].forEach(face => {
      var listItem.textContent = faces;
      faceList.appendChild(listItem);
    });
  });
}
