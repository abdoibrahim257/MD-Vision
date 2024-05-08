import React, { useEffect, useReducer } from 'react'
import '../styles/Chat.css'
import NavBar from './navbar.js'
import robot from '../assets/loveBot.svg'

import { WindupChildren } from "windups";

import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message
} from "@chatscope/chat-ui-kit-react";
import { redirect } from 'react-router-dom';

const initialState = { index: 0 };

function reducer(state, action) {
  switch (action.type) {
    case 'increment':
      return { index: state.index + 1 };
    // Add more cases for other actions
    default:
      throw new Error();
  }
}

const ChatPage = () => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const incrementIndex = () => {
    dispatch({ type: 'increment' });
  };

  // const idx = useRef(0);
  const [handleBtn, setHandleBtn] = React.useState(false)
  // const [idx, setIdx] = React.useState(0)
  
  // const [messages, setMessages] = React.useState([])
  const [messages, setMessages] = React.useState({})

  //   message: "Hello, I'm Maven, your personal health assistant. How can I help you today?",
  //   sender: 'Maven'
  
  const [diagnosed, setDiagnosed] = React.useState(false)
  // const [typing, setTyping] = React.useState(false)

  const currentLink = window.location.href

  // const typeWritter = (index, msg, msgIndex) => {
  //   setTimeout(() => {
  //     //get index of the message 
  //     var newMessages = messages;
  //     newMessages[index] = {message: msg.substr(0,msgIndex), sender: 'Maven'} 
  //     // setMessages([...newMessages])
  //     setMessages(newMessages)
  //     if ( msgIndex < msg.length) {
  //       typeWritter(index, msg, msgIndex+1)
  //     }
  //   }, 23)
  // }
  const typeWritter = (key, msg, msgIndex) => {
    setTimeout(() => {
      // Update the state
      setMessages((oldMessages) => {
        // Create a copy of the messages object
        var newMessages = {...oldMessages};
        // Update the message at the given key
        newMessages[key] = {...newMessages[key], message: msg.substr(0, msgIndex)};
        return newMessages;
      });
      // If there are more characters in the message, call typeWritter again
      if (msgIndex < msg.length) {
        typeWritter(key, msg, msgIndex + 1);
      }
    }, 23);
  };
   
  const fetchFirstMessage = async () => {
    let symptom = currentLink.split('/').pop()
    const response = await fetch('http://127.0.0.1:8000/maven/'+symptom)
    const data = await response.json()
    
    // console.log(data)
    
    // setTyping(true);

    const message = {
      // message: data.Question,
      message: "",
      sender: 'Maven'
    }

    // const newMessages = [...messages, message]

    // setMessages((oldmessages) => {
    //   let newMsgs = [...oldmessages]
    //   newMsgs.push(message)
    //   return newMsgs;
    // })
    setMessages((oldMessages = {}) => {
      var newMessages = {...oldMessages};
      let newKey = state.index; // get the next key
      // console.log(newKey)
      newMessages[newKey] = message;
      return newMessages;
    });
    

    typeWritter(state.index, data.Question, 0);
    incrementIndex();
    // console.log("BEFORREE"+idx);
    // setIdx(idx+1);
    // console.log("AFTERRR"+idx);
    console.log("IDXXX AFTER FIRSTQ"+state.index)

  }
  useEffect(() => {
    fetchFirstMessage()
  }, [])
  


  async function fetchNextQuestion (answer) {
    let symptom = currentLink.split('/').pop();
    let url = 'http://127.0.0.1:8000/maven/'+symptom;

    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json', //specify the content type
      },
      body: JSON.stringify({ans: answer}),
    }).then(response => response.json())
    .then(data => {
      const Q = data.Question
      // check if the word Diagnosis: is in the question
      if (Q.includes('Diagnosis:')) {
        setDiagnosed(true)
      }
      // setTyping(true)
      const m = {
        // message: Q,
        message: "",
        sender: 'Maven'
      }
      
      // setIdx(indx)
      // console.log(indx)
      // console.log(idx)
      // const newM = [...updateMsgs, m];
      // setMessages(newM)
      // setMessages((oldmessages) => {
      //   let newMsgs = [...oldmessages]
      //   newMsgs.push(m)
      //   return newMsgs;
      // })

      setMessages((oldMessages = {}) => {
        var newMessages = {...oldMessages};
        let newKey = state.index // get the next key
        // console.log(newKey)
        // console.log(newMessages)
        newMessages[newKey] = m;
        return newMessages;
      });
      
      //get index of last message 
      // console.log(Q)
      // console.log(idx)
      console.log(messages)
      typeWritter(state.index, Q,0)
      incrementIndex();
      // idx.current+=1;
      // const nID = indx + 1;
      // setIdx(nID)
      // console.log("INDDXXXX="+inx)
      // console.log("IDXXX AFTER NEXTQ"+idx)
    })
    //get data from promise 
  }

  const handleYes = async () => {
    setHandleBtn(true);

    const newMessage = {
      message: "Yes",
      sender: 'user',
      direction: "outgoing"
    }
    
    //before adding the answer I want to set the previous question to be answered
    // const lastQ = messages[messages.length - 1]
    // const updatedLastQ = {
    //   ...lastQ,
    //   status: 'answered'
    // }

    // const newMessages = messages.slice(0, messages.length - 1)
    // const update = [...newMessages, updatedLastQ, newMessage]
    setMessages((oldMessages = {}) => {
      var newMessages = {...oldMessages};
      // let newKey = Object.keys(newMessages).length; // get the next key
      let newKey = state.index; // get the next key MABYT8RSHHHH
      console.log(newKey)
      newMessages[newKey-2] = {...newMessages[newKey-2], status: "answered"};
      newMessages[newKey-1] = newMessage;
      console.log(newMessages)
      // console.log("BEFORE YES"+idx);
      // setIdx(newKey)
      return newMessages;
    })
    // console.log("IDX AFTER YES: " + idx)
    // console.log("AFTER YES"+idx);
    // console.log(messages)
    incrementIndex();
    fetchNextQuestion('yes')


    // setMessages((oldmessages) => {
    //   let newMsgs = [...oldmessages]
    //   // newMsgs.slice(0, messages.length - 1)
    //   // newMsgs.push(updatedLastQ)
    //   // console.log(idx-1)
    //   if (newMsgs[idx - 1]) {
    //     newMsgs[idx - 1] ={...newMsgs[idx - 1], status: "answered"}
    //   }
    //   newMsgs.push(newMessage)
    //   // setIdx((idx) => idx+1);
    //   console.log(newMsgs)
    //   return newMsgs;
    // })

    // )
    
    // setIdx(idx+1)
    
    // setMessages(update);
    
    // because the function will update the messages state with the new message 
    
  }

  const handleNo = async () => {
    setHandleBtn(true);

    const newMessage = {
      message: "No",
      sender: 'user',
      direction: "outgoing"
    }
    
    //before adding the answer I want to set the previous question to be answered
    // const lastQ = messages[messages.length - 1]
    // const updatedLastQ = {
    //   ...lastQ,
    //   status: 'answered'
    // }

    // const newMessages = messages.slice(0, messages.length - 1)
    // const update = [...newMessages, updatedLastQ, newMessage]
    setMessages((oldMessages = {}) => {
      var newMessages = {...oldMessages};
      // let newKey = Object.keys(newMessages).length; // get the next key
      let newKey = state.index; // get the next key MABYT8RSHHHH
      console.log(newKey)
      newMessages[newKey-2] = {...newMessages[newKey-2], status: "answered"};
      newMessages[newKey-1] = newMessage;
      console.log(newMessages)
      // console.log("BEFORE YES"+idx);
      // setIdx(newKey)
      return newMessages;
    })
    // console.log("IDX AFTER YES: " + idx)
    // console.log("AFTER YES"+idx);
    // console.log(messages)
    incrementIndex();
    fetchNextQuestion('no')
  }
 

  return (
    <div>
        <NavBar sticky={1}/>
        <div className='chat-section'>
          <div className='warning-message chat-content'>
            <p> <span>Warning: </span> This tool is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a life-threatening emergency that requires immediate attention please call 123 or the number for your local emergency service.</p>
          </div>
          <div className='interactice-section'> 
          </div>

          <div className='interactice-section'>
            <MainContainer>
              <ChatContainer>
                <MessageList>
                    {
                      Object.keys(messages).map((key) => {
                      const message = messages[key];
                      {/* console.log(message.status) */}
                      if (message.sender === 'Maven' && !message.status) {
                        return (
                          <div key={key}>
                            <div className='message-wrapper'>
                              <img src={robot} alt='robot' className='robot'/>
                              <Message className = "message" model={message}/>
                            </div>
                            <div className={diagnosed? 'Hidden' : 'answer-wrapper'}>
                              <button className='btn yes' onClick={handleYes}>Yes</button>
                              <button className='btn no' onClick={handleNo}>No</button>
                            </div>
                          </div>
                        );
                      }
                      else if (message.status && message.status === "answered" && message.sender === 'Maven') {
                        {/* console.log("YOOOOOOOOOOOOOOOOOOOOOOOOOO") */}
                        return (
                            <div key={key} className='message-wrapper'>
                              <img src={robot} alt='robot' className='robot'/>
                              <Message className = "message" model={message}/>
                            </div>
                        );
                      }
                      else {
                        return (
                          <Message className = "answer-message" key={key} model={message}/>
                        );
                      }
                    })
                    }
                </MessageList>
              </ChatContainer>
            </MainContainer>
          </div>


        </div>
  
    </div>
  )
}

export default ChatPage