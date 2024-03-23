class Chatbox {
    constructor() {
        this.args = {
            openButton : document.querySelector('.chatbox__button'),
            chatBox : document.querySelector('.chatbox__support'),
            sendButton : document.querySelector('.send__button'),
            billButton : document.querySelector('.bill')
        }

        this.state = false;
        this.messages = [];
    };

    display() {
        const {openButton, chatBox, sendButton,billButton} = this.args;
        
        openButton.addEventListener('click', () => this.toggleState(chatBox))
        sendButton.addEventListener('click', () => this.onSendButton(chatBox))
        billButton.addEventListener('click', () => this.onbillButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener('keyup', ({key}) => {
            if(key === "Enter"){
                this.onSendButton(chatBox)
            }
        })
        document.querySelector(".send__button").addEventListener("keydown", function(event) {
            if (event.keyCode === 13) { // Check if Enter key was pressed
                event.preventDefault(); // Prevent the default action (e.g., newline in textarea)
                submitForm();
            }
        });
    }

    toggleState(chatBox) {
        this.state = !this.state;

        if(this.state){
            chatBox.classList.add('chatbox--active')
        } else {
            chatBox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatBox) {
        var textField = chatBox.querySelector('input');
        let text1 = textField.value
        if(text1 === ""){
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);
        this.updateChatText(chatBox)
        textField.value = ''

        fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ content: text1 })
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: "Bot", message: r.message};
            this.messages.push(msg2);
            console.log(msg2)
            this.updateChatText(chatBox)
        }).catch((error) => {
            console.log('Error:', error);
            this.updateChatText(chatBox)
        });
    }
    

    updateChatText(chatBox) {
        var html = '';

        this.messages.slice().reverse().forEach(function(item) {
            if(item.name === "Bot")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });

        const chatmessage = chatBox.querySelector('.chatbox__messages')
        chatmessage.innerHTML = html;
    }
}
function submitForm(event) {
    event.preventDefault();
    
    var formData = {
        name: document.getElementById("name").value,
        age: document.getElementById("age").value,
        gender: document.querySelector('input[name="gender"]:checked').value
    };

    // Convert form data to JSON
    var jsonData = JSON.stringify(formData);
    console.log(jsonData);
    // Here, you can send the JSON data to a server using AJAX or perform any other action
}
function validateForm() {
    var name = document.getElementById("name").value;
    var email = document.getElementById("email").value;
    var password = document.getElementById("password").value;
    var isValid = true;

    if (name === "") {
        document.getElementById("name").classList.add("error");
        isValid = false;
    } else {
        document.getElementById("name").classList.remove("error");
    }

    if (email === "") {
        document.getElementById("email").classList.add("error");
        isValid = false;
    } else {
        document.getElementById("email").classList.remove("error");
    }

    if (password === "") {
        document.getElementById("password").classList.add("error");
        isValid = false;
    } else {
        document.getElementById("password").classList.remove("error");
    }

    return isValid;
}

const chatbox = new Chatbox();
chatbox.display();