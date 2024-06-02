mybot.js
function processUserInput() {
	var user_input = document.getElementById("inputField").value;
	const chatContainer = document.getElementById('chat-container');
	const messageDiv = document.createElement('div');
	messageDiv.innerHTML = `<div class='font-weight-bolder text-right alert alert-dark shadow'> 'User': ${user_input} </div>`;
	chatContainer.appendChild(messageDiv);
	// frappe.throw(user_input);
	// document.getElementById("response").innerText = user_input;
	frappe.call({
			method: 'hexbot.app.get_chat_response',
			args: {
					'user_input': user_input
			},
			callback: function(response) {
					// const messageDiv = document.createElement('div')
					// messageDiv.textContent = `'User': ${user_input}`;
					const responseDiv = document.createElement('div');
					responseDiv.innerHTML = `<div class='font-weight-bold alert alert-light shadow bg-white rounded'> 'Bot': ${response.message}</div>`;
					chatContainer.appendChild(responseDiv);
					// var user_text = user_input;
					// var processed_response = response.message;
					// document.getElementById("response").innerText = processed_response;
				
			}
	})


}

const clearButton = document.getElementById('clearButton');
const listElement = document.getElementById('list');

clearButton.addEventListener('click', function() {
    while (listElement.firstChild) {
        listElement.removeChild(listElement.firstChild);
    }
	frappe.call({
		method: 'hexbot.www.hexbot.clear_history',  // Path to your Python function
		callback: function(response) {
			const listElement = document.getElementById('list');
			response.message.forEach(item => {
				const li = document.createElement('li');
				li.textContent = item;
				listElement.appendChild(li);
			});
		}
	});
});



// Get a reference to the button element
var reloadButton = document.getElementById('reloadButton');

// Add a click event listener to the button
reloadButton.addEventListener('click', function() {
	// Reload the page
	location.reload();
});
   