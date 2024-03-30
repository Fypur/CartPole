using AI;
using Fiourp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CartPole
{
    public class Env2 : Gym
    {
        public Cart Cart;
        public Pole Pole;
        private int maxDistanceCar = 200;
        protected override string rewardGraphSaveLocation { get => System.Environment.CurrentDirectory + "/rewardGraph.png"; }

        public Env2(DeepQAgent2 agent) : base(agent, 1000, 5000)
        {
            Cart = new Cart();
            Pole = new Pole(Cart);

            Reset();
        }

        protected override bool Done()
            => Cart.MiddlePos.X > Engine.ScreenSize.X / 2 + maxDistanceCar || Cart.MiddlePos.X < Engine.ScreenSize.X / 2 - maxDistanceCar || Pole.Pos.Y > Cart.Pos.Y - 200 + 60 || EpisodeStep >= 1000;

        protected override float[] GetState()
        {
            return new float[4]
            {
                Normalize(Cart.MiddlePos.X - Engine.ScreenSize.X / 2, -maxDistanceCar, maxDistanceCar),
                Normalize(Cart.Velocity, -20, 20),
                Normalize(Pole.GetAngle(out _), -0.72f, 0.72f),
                Normalize(Pole.Pos.X - Cart.MiddlePos.X, -130, 130)
            };
        }

        protected override float UpdateAndReward(int action)
        {
            Cart.Update(action);
            Pole.Update();

            return 1;
        }

        protected override void Reset()
        {
            if (EpisodeStep > 100)
                Console.ForegroundColor = ConsoleColor.Green;
            else if (EpisodeStep > 70)
                Console.ForegroundColor = ConsoleColor.Yellow;
            else
                Console.ForegroundColor = ConsoleColor.Red;

            Console.WriteLine($"Episode {Main.episode}, Score {EpisodeStep}, Epsilon: {Agent.Epsilon}, Beta: {Agent.ReplayBuffer.Beta}");
            Console.ForegroundColor = ConsoleColor.Gray;

            Main.episode++;

            Cart.Reset();
            Pole.Reset();
        }

        public override void Render()
        {
            Cart.Render();
            Pole.Render();
        }

        public float Normalize(float value, float min, float max)
        => (value - min) / (max - min);
    }
}
